// xstate_sigproof: what a thread of an x86-64 macOS guest keeps across a
// signal handler and across thread_get_state/thread_set_state, with RgfxGate's
// xstate route (src/RgfxGateXstate.cpp) on or off. Written from scratch; it
// runs in the guest, on each macOS release, from the walk's sigproof step.
//
//   xstate_sigproof [SECONDS [THREADS]]   (default 60 s, 2 x hw.ncpu threads)
//
// Lines it prints:
//   env ...          what the main thread runs at (xcr0), what apps are told
//                    (leaf 7's avx512f, leaf 0xD's eax, hw.optional.avx512f),
//                    where XSAVE puts components 5-7 (leaf 0xD, sub-leaves
//                    5-7), whether the route promoted this thread, the release
//                    and the CPU count
//   i-signal ...     (i), signals: x87 (8 loaded registers, control word,
//                    tags), SSE (xmm0-15, MXCSR) and the ymm upper halves
//                    (ymmh0-15) of a thread that never runs EVEX, loaded from a
//                    pattern, then a ud2 whose SIGILL handler checks the signal
//                    frame against the pattern, overwrites every one of those
//                    registers and steps over the ud2; after the handler, the
//                    registers are stored and compared with the pattern. The
//                    frame's flavor (mcsize: avx, avx512, and their _full forms)
//                    is counted.
//   i-thread-state . (i), thread state: a second thread holds a pattern in its
//                    registers and spins (integer only); the main thread
//                    suspends it and, per case, gets and sets its state
//                    (x86_AVX_STATE64 and x86_AVX512_STATE64, unchanged, then
//                    changed), resumes it, and the thread stores and compares.
//                    Where the route copies at CPUID's offsets (promoted, and
//                    the offsets not Intel's), the AVX-512 flavor for that other
//                    thread must be refused; elsewhere a refusal is recorded.
//   i-load ...       (i) under load: THREADS threads for SECONDS, each round a
//                    new pattern, alternately a ud2 (a synchronous SIGILL) and
//                    an integer spin while a sender thread sends SIGUSR1 to
//                    every thread in turn (asynchronous signals; a frame whose
//                    rip is in the spin is checked against the pattern), with
//                    more threads than CPUs, so rounds are preempted too; a
//                    churn thread creates short-lived threads that each run one
//                    round (each new thread's save area). Counts rounds,
//                    signals, frame checks and mismatches, and each thread's
//                    xcr0 after its first round.
//   ii ...           (ii), a thread that runs EVEX without checking the feature
//                    (zmm0's upper 256 bits, zmm16, k3 loaded), across a SIGILL
//                    handler that leaves those registers alone, and one that
//                    overwrites them: each must come back, and the frame must
//                    be the AVX-512 one and hold them.
//   ii-self ...      (ii), the same thread's own state: zmm16 and k3 loaded,
//                    then thread_get_state of itself with x86_AVX512_STATE64
//                    must read them, and thread_set_state of itself with new
//                    values must leave those in the registers.
//   result PASS|FAIL (i) and (ii); exit status 0 on PASS. Where EVEX is refused
//                    (#UD: the guest has no AVX-512 state), (ii) prints so and
//                    passes.
//
// Built on a Mac by `make -C guest/rgfxgate sigproof`.
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <unistd.h>

// The registers a pattern sets. Offsets are fixed: the assembly below uses
// them.
struct pat {
	uint8_t ymm[16][32]; // 0: ymm0-15, low 16 bytes the xmm register, high 16 its upper half
	uint8_t st[8][16];   // 512: ST(0)-ST(7), 10 bytes each (80-bit), 16 apart
	uint32_t mxcsr;      // 640
	uint16_t fcw;        // 644
	uint16_t pad;
	uint8_t pad2[8];
};
_Static_assert(offsetof(struct pat, st) == 512, "st");
_Static_assert(offsetof(struct pat, mxcsr) == 640, "mxcsr");
_Static_assert(offsetof(struct pat, fcw) == 644, "fcw");

// The EVEX case: zmm0 (64 bytes), zmm16 (64), k3 (8).
struct evex {
	uint8_t zmm0[64];
	uint8_t zmm16[64];
	uint64_t k3;
	uint64_t pad;
};
_Static_assert(offsetof(struct evex, zmm16) == 64, "zmm16");
_Static_assert(offsetof(struct evex, k3) == 128, "k3");

#define LOAD_PAT                                                                                   \
	"vmovdqu 0(%rdi), %ymm0\n vmovdqu 32(%rdi), %ymm1\n vmovdqu 64(%rdi), %ymm2\n"                \
	"vmovdqu 96(%rdi), %ymm3\n vmovdqu 128(%rdi), %ymm4\n vmovdqu 160(%rdi), %ymm5\n"             \
	"vmovdqu 192(%rdi), %ymm6\n vmovdqu 224(%rdi), %ymm7\n vmovdqu 256(%rdi), %ymm8\n"            \
	"vmovdqu 288(%rdi), %ymm9\n vmovdqu 320(%rdi), %ymm10\n vmovdqu 352(%rdi), %ymm11\n"          \
	"vmovdqu 384(%rdi), %ymm12\n vmovdqu 416(%rdi), %ymm13\n vmovdqu 448(%rdi), %ymm14\n"         \
	"vmovdqu 480(%rdi), %ymm15\n"                                                                 \
	"ldmxcsr 640(%rdi)\n fninit\n fldcw 644(%rdi)\n"                                              \
	"fldt 624(%rdi)\n fldt 608(%rdi)\n fldt 592(%rdi)\n fldt 576(%rdi)\n"                         \
	"fldt 560(%rdi)\n fldt 544(%rdi)\n fldt 528(%rdi)\n fldt 512(%rdi)\n"
#define STORE_PAT                                                                                  \
	"vmovdqu %ymm0, 0(%rsi)\n vmovdqu %ymm1, 32(%rsi)\n vmovdqu %ymm2, 64(%rsi)\n"                \
	"vmovdqu %ymm3, 96(%rsi)\n vmovdqu %ymm4, 128(%rsi)\n vmovdqu %ymm5, 160(%rsi)\n"             \
	"vmovdqu %ymm6, 192(%rsi)\n vmovdqu %ymm7, 224(%rsi)\n vmovdqu %ymm8, 256(%rsi)\n"            \
	"vmovdqu %ymm9, 288(%rsi)\n vmovdqu %ymm10, 320(%rsi)\n vmovdqu %ymm11, 352(%rsi)\n"          \
	"vmovdqu %ymm12, 384(%rsi)\n vmovdqu %ymm13, 416(%rsi)\n vmovdqu %ymm14, 448(%rsi)\n"         \
	"vmovdqu %ymm15, 480(%rsi)\n"                                                                 \
	"stmxcsr 640(%rsi)\n fnstcw 644(%rsi)\n"                                                      \
	"fstpt 512(%rsi)\n fstpt 528(%rsi)\n fstpt 544(%rsi)\n fstpt 560(%rsi)\n"                     \
	"fstpt 576(%rsi)\n fstpt 592(%rsi)\n fstpt 608(%rsi)\n fstpt 624(%rsi)\n"                     \
	"fninit\n vzeroupper\n"

// sp_round(in, out, spin): loads in, then a ud2 (spin 0: the SIGILL handler
// steps over it) or an integer loop of spin passes, then stores the registers
// to out.
// sp_ts_round(in, out, flag): loads in, sets *flag to 1, spins until *flag is
// 2, stores to out.
// sp_evex_round(in, out): loads zmm0, zmm16 and k3, a ud2, stores them;
// sp_evex_bail returns from it (an EVEX #UD).
// sp_self_round(in, out, fn): loads zmm16 and k3 from in, calls fn, stores
// zmm16 and k3 to out. fn is C code built for x86-64 without AVX-512, which
// leaves zmm16-31 and k0-7 alone (zmm0-15's upper halves do not survive a
// call: VEX code and vzeroupper clear them).
void sp_round(const struct pat *in, struct pat *out, long spin);
void sp_ts_round(const struct pat *in, struct pat *out, volatile int *flag);
void sp_evex_round(const struct evex *in, struct evex *out);
void sp_self_round(const struct evex *in, struct evex *out, void (*fn)(void));
void sp_clobber_evex(void); // overwrites zmm0, zmm16 and k3
extern char sp_trap[], sp_spin_begin[], sp_spin_end[], sp_evex_round_start[], sp_evex_trap[],
    sp_evex_end[], sp_evex_bail[];
__asm__(".text\n .p2align 4\n .globl _sp_round\n _sp_round:\n" LOAD_PAT
        "test %rdx, %rdx\n jnz 2f\n"
        ".globl _sp_trap\n _sp_trap:\n ud2\n jmp 3f\n"
        "2:\n .globl _sp_spin_begin\n _sp_spin_begin:\n dec %rdx\n jnz _sp_spin_begin\n"
        ".globl _sp_spin_end\n _sp_spin_end:\n"
        "3:\n" STORE_PAT "ret\n"
        ".p2align 4\n .globl _sp_ts_round\n _sp_ts_round:\n" LOAD_PAT
        "movl $1, (%rdx)\n"
        "4:\n cmpl $2, (%rdx)\n jne 4b\n" STORE_PAT "ret\n"
        ".p2align 4\n .globl _sp_evex_round\n .globl _sp_evex_round_start\n"
        " _sp_evex_round:\n _sp_evex_round_start:\n"
        "vmovdqu64 (%rdi), %zmm0\n vmovdqu64 64(%rdi), %zmm16\n kmovq 128(%rdi), %k3\n"
        ".globl _sp_evex_trap\n _sp_evex_trap:\n ud2\n"
        "vmovdqu64 %zmm0, (%rsi)\n vmovdqu64 %zmm16, 64(%rsi)\n kmovq %k3, 128(%rsi)\n"
        ".globl _sp_evex_end\n _sp_evex_end:\n vzeroupper\n ret\n"
        ".globl _sp_evex_bail\n _sp_evex_bail:\n vzeroupper\n ret\n"
        ".p2align 4\n .globl _sp_self_round\n _sp_self_round:\n"
        "pushq %rbx\n movq %rsi, %rbx\n"
        "vmovdqu64 64(%rdi), %zmm16\n kmovq 128(%rdi), %k3\n"
        "callq *%rdx\n"
        "vmovdqu64 %zmm16, 64(%rbx)\n kmovq %k3, 128(%rbx)\n"
        "popq %rbx\n vzeroupper\n ret\n"
        ".p2align 4\n .globl _sp_clobber_evex\n _sp_clobber_evex:\n"
        "vpternlogd $0xff, %zmm0, %zmm0, %zmm0\n vpternlogd $0xff, %zmm16, %zmm16, %zmm16\n"
        "kxnorq %k3, %k3, %k3\n vzeroupper\n ret\n");

// --- patterns ---------------------------------------------------------------
static uint64_t xs(uint64_t *s) {
	uint64_t x = *s;
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	return *s = x;
}
static void fill(struct pat *p, uint64_t *seed) {
	memset(p, 0, sizeof *p);
	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 32; j += 8) {
			uint64_t v = xs(seed);
			memcpy(&p->ymm[i][j], &v, 8);
		}
	// A normal 80-bit value: explicit integer bit, exponent 1..0x7ffe.
	for (int i = 0; i < 8; i++) {
		uint64_t m = xs(seed) | (1ull << 63);
		uint16_t e = (uint16_t)(1 + xs(seed) % 0x7ffe) | (uint16_t)((xs(seed) & 1) << 15);
		memcpy(&p->st[i][0], &m, 8);
		memcpy(&p->st[i][8], &e, 2);
	}
	uint64_t r = xs(seed);
	// MXCSR: exceptions masked, rounding (13-14), flush to zero (15) and
	// denormals are zero (6) from the pattern.
	p->mxcsr = 0x1f80 | (uint32_t)((r & 3) << 13) | (uint32_t)(((r >> 2) & 1) << 15) |
	           (uint32_t)(((r >> 3) & 1) << 6);
	// FCW: exceptions masked (0-5, and 6 as the CPU keeps it), precision 24, 53
	// or 64 bits, rounding from the pattern.
	static const uint16_t pc[3] = {0, 2, 3};
	p->fcw = (uint16_t)(0x007f | (pc[(r >> 4) % 3] << 8) | (((r >> 6) & 3) << 10));
}
static void fill_evex(struct evex *e, uint64_t *seed) {
	memset(e, 0, sizeof *e);
	for (int j = 0; j < 64; j += 8) {
		uint64_t v = xs(seed);
		memcpy(e->zmm0 + j, &v, 8);
		v = xs(seed);
		memcpy(e->zmm16 + j, &v, 8);
	}
	e->k3 = xs(seed);
}
// The fields of out that differ from in, as a mask: 1 ymm low (xmm), 2 ymm
// high, 4 x87 registers, 8 MXCSR, 16 FCW.
static int differ(const struct pat *in, const struct pat *out) {
	int d = 0;
	for (int i = 0; i < 16; i++) {
		if (memcmp(in->ymm[i], out->ymm[i], 16))
			d |= 1;
		if (memcmp(in->ymm[i] + 16, out->ymm[i] + 16, 16))
			d |= 2;
	}
	for (int i = 0; i < 8; i++)
		if (memcmp(in->st[i], out->st[i], 10))
			d |= 4;
	if (in->mxcsr != out->mxcsr)
		d |= 8;
	if (in->fcw != out->fcw)
		d |= 16;
	return d;
}

// --- the frame and the thread state -----------------------------------------
// The AVX part both state flavors share, at the AVX-512 struct's offsets (the
// AVX struct's are the same).
typedef struct __darwin_x86_avx512_state64 fs512_t;
_Static_assert(offsetof(struct __darwin_x86_avx_state64, __fpu_ymmh15) ==
                   offsetof(fs512_t, __fpu_ymmh15),
               "avx");
// The fields of a frame or thread state (AVX part) that differ from p, as
// differ() counts them; 32 the tag word.
static int fs_differ(const void *fs, const struct pat *p) {
	const uint8_t *b = fs;
	int d = 0;
	for (int i = 0; i < 16; i++) {
		if (memcmp(b + offsetof(fs512_t, __fpu_xmm0) + 16 * i, p->ymm[i], 16))
			d |= 1;
		if (memcmp(b + offsetof(fs512_t, __fpu_ymmh0) + 16 * i, p->ymm[i] + 16, 16))
			d |= 2;
	}
	for (int i = 0; i < 8; i++)
		if (memcmp(b + offsetof(fs512_t, __fpu_stmm0) + 16 * i, p->st[i], 10))
			d |= 4;
	uint32_t mx;
	memcpy(&mx, b + offsetof(fs512_t, __fpu_mxcsr), 4);
	if (mx != p->mxcsr)
		d |= 8;
	uint16_t cw;
	memcpy(&cw, b + offsetof(fs512_t, __fpu_fcw), 2);
	if (cw != p->fcw)
		d |= 16;
	if (b[offsetof(fs512_t, __fpu_ftw)] != 0xff)
		d |= 32;
	return d;
}
static void fs_put(void *fs, const struct pat *p) {
	uint8_t *b = fs;
	for (int i = 0; i < 16; i++) {
		memcpy(b + offsetof(fs512_t, __fpu_xmm0) + 16 * i, p->ymm[i], 16);
		memcpy(b + offsetof(fs512_t, __fpu_ymmh0) + 16 * i, p->ymm[i] + 16, 16);
	}
	for (int i = 0; i < 8; i++)
		memcpy(b + offsetof(fs512_t, __fpu_stmm0) + 16 * i, p->st[i], 10);
	memcpy(b + offsetof(fs512_t, __fpu_mxcsr), &p->mxcsr, 4);
	memcpy(b + offsetof(fs512_t, __fpu_fcw), &p->fcw, 2);
	b[offsetof(fs512_t, __fpu_ftw)] = 0xff;
}

enum { MC_AVX, MC_AVX_FULL, MC_512, MC_512_FULL, MC_OTHER, MC_N };
static const char *const mc_name[MC_N] = {"avx", "avx_full", "avx512", "avx512_full", "other"};
static int mc_kind(size_t sz, size_t *fs_off) {
	if (sz == sizeof(struct __darwin_mcontext_avx64)) {
		*fs_off = offsetof(struct __darwin_mcontext_avx64, __fs);
		return MC_AVX;
	}
	if (sz == sizeof(struct __darwin_mcontext_avx64_full)) {
		*fs_off = offsetof(struct __darwin_mcontext_avx64_full, __fs);
		return MC_AVX_FULL;
	}
	if (sz == sizeof(struct __darwin_mcontext_avx512_64)) {
		*fs_off = offsetof(struct __darwin_mcontext_avx512_64, __fs);
		return MC_512;
	}
	if (sz == sizeof(struct __darwin_mcontext_avx512_64_full)) {
		*fs_off = offsetof(struct __darwin_mcontext_avx512_64_full, __fs);
		return MC_512_FULL;
	}
	return MC_OTHER;
}

// --- the handlers -----------------------------------------------------------
struct counts {
	atomic_ulong rounds, sync, async, frame_checks, frame_bad, bad, bad_bits;
	atomic_ulong mc[MC_N];
};
static struct counts C;
// The pattern the thread's registers hold, while they hold it.
static __thread const struct pat *sp_cur;
static __thread int sp_frame_bits;
// The EVEX case's state.
static const struct evex *ev_cur;
static int ev_write; // the handler overwrites zmm0, zmm16 and k3
static int ev_frame_kind = -1, ev_frame_zmmh0 = -1, ev_frame_zmm16 = -1, ev_frame_k3 = -1, ev_ud;

static void clobber(void) {
	__asm__ volatile("vpcmpeqd %%ymm0, %%ymm0, %%ymm0\n vpcmpeqd %%ymm1, %%ymm1, %%ymm1\n"
	                 "vpcmpeqd %%ymm2, %%ymm2, %%ymm2\n vpcmpeqd %%ymm3, %%ymm3, %%ymm3\n"
	                 "vpcmpeqd %%ymm4, %%ymm4, %%ymm4\n vpcmpeqd %%ymm5, %%ymm5, %%ymm5\n"
	                 "vpcmpeqd %%ymm6, %%ymm6, %%ymm6\n vpcmpeqd %%ymm7, %%ymm7, %%ymm7\n"
	                 "vpcmpeqd %%ymm8, %%ymm8, %%ymm8\n vpcmpeqd %%ymm9, %%ymm9, %%ymm9\n"
	                 "vpcmpeqd %%ymm10, %%ymm10, %%ymm10\n vpcmpeqd %%ymm11, %%ymm11, %%ymm11\n"
	                 "vpcmpeqd %%ymm12, %%ymm12, %%ymm12\n vpcmpeqd %%ymm13, %%ymm13, %%ymm13\n"
	                 "vpcmpeqd %%ymm14, %%ymm14, %%ymm14\n vpcmpeqd %%ymm15, %%ymm15, %%ymm15\n"
	                 "fninit\n fldz\n fldz\n fldz\n fldz\n fldz\n fldz\n fldz\n fldz\n fninit\n"
	                 "pushq $0x1f80\n ldmxcsr (%%rsp)\n addq $8, %%rsp\n vzeroupper\n"
	                 :
	                 :
	                 : "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8",
	                   "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "st",
	                   "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)", "memory");
}
// A byte compare that calls nothing and is not vectorised: the EVEX case's
// handler must leave the vector registers as they are (a library call could
// clear their upper halves).
static int same(const volatile uint8_t *a, const uint8_t *b, int n) {
	int d = 0;
	for (int i = 0; i < n; i++)
		d |= a[i] ^ b[i];
	return d == 0;
}

static void frame_check(const ucontext_t *uc, const struct pat *p) {
	size_t off = 0;
	int k = mc_kind(uc->uc_mcsize, &off);
	atomic_fetch_add_explicit(&C.mc[k], 1, memory_order_relaxed);
	atomic_fetch_add_explicit(&C.frame_checks, 1, memory_order_relaxed);
	if (k == MC_OTHER) {
		atomic_fetch_add_explicit(&C.frame_bad, 1, memory_order_relaxed);
		return;
	}
	int d = fs_differ((const char *)uc->uc_mcontext + off, p);
	if (d) {
		atomic_fetch_add_explicit(&C.frame_bad, 1, memory_order_relaxed);
		sp_frame_bits |= d;
	}
}

static void on_signal(int sig, siginfo_t *si, void *ctx) {
	(void)si;
	ucontext_t *uc = ctx;
	uint64_t *rip = &uc->uc_mcontext->__ss.__rip;
	char *ip = (char *)*rip;
	if (sig == SIGILL && ip == sp_trap) {
		if (sp_cur)
			frame_check(uc, sp_cur);
		atomic_fetch_add_explicit(&C.sync, 1, memory_order_relaxed);
		*rip += 2;
	} else if (sig == SIGILL && ip == sp_evex_trap) {
		size_t off = 0;
		int k = mc_kind(uc->uc_mcsize, &off);
		ev_frame_kind = k;
		if (k == MC_512 || k == MC_512_FULL) {
			const uint8_t *b = (const uint8_t *)uc->uc_mcontext + off;
			ev_frame_zmmh0 = same(b + offsetof(fs512_t, __fpu_zmmh0), ev_cur->zmm0 + 32, 32);
			ev_frame_zmm16 = same(b + offsetof(fs512_t, __fpu_zmm16), ev_cur->zmm16, 64);
			ev_frame_k3 = same(b + offsetof(fs512_t, __fpu_k3), (const uint8_t *)&ev_cur->k3, 8);
		}
		if (ev_write)
			sp_clobber_evex();
		*rip += 2;
		return; // no clobber(): its VEX writes clear zmm0-15's upper halves
	} else if (sig == SIGILL && ip >= sp_evex_round_start && ip < sp_evex_end) {
		ev_ud = 1; // EVEX refused (#UD): return from the case
		*rip = (uint64_t)(uintptr_t)sp_evex_bail;
		return;
	} else if (sig == SIGUSR1) {
		atomic_fetch_add_explicit(&C.async, 1, memory_order_relaxed);
		if (sp_cur && ip >= sp_spin_begin && ip < sp_spin_end)
			frame_check(uc, sp_cur);
	} else {
		// A SIGILL anywhere else: not ours.
		fprintf(stderr, "unexpected signal %d at %p\n", sig, (void *)ip);
		_exit(3);
	}
	clobber();
}

// --- (i) signals: one thread ------------------------------------------------
static int one_round(uint64_t *seed, long spin, int *bits) {
	struct pat in, out;
	fill(&in, seed);
	memset(&out, 0, sizeof out);
	sp_cur = &in;
	atomic_signal_fence(memory_order_seq_cst);
	sp_round(&in, &out, spin);
	atomic_signal_fence(memory_order_seq_cst);
	sp_cur = NULL;
	int d = differ(&in, &out);
	*bits |= d;
	return d != 0;
}
static const char *bits_str(int d, char *buf) {
	if (!d)
		return "ok";
	buf[0] = 0;
	if (d & 1)
		strcat(buf, "xmm,");
	if (d & 2)
		strcat(buf, "ymmh,");
	if (d & 4)
		strcat(buf, "x87,");
	if (d & 8)
		strcat(buf, "mxcsr,");
	if (d & 16)
		strcat(buf, "fcw,");
	if (d & 32)
		strcat(buf, "ftw,");
	buf[strlen(buf) - 1] = 0;
	return buf;
}

// --- (i) thread state -------------------------------------------------------
struct ts_arg {
	struct pat in, out;
	volatile int flag;
};
static void *ts_main(void *a) {
	struct ts_arg *t = a;
	sp_ts_round(&t->in, &t->out, &t->flag);
	return NULL;
}
// case 0: get and set x86_AVX_STATE64 unchanged; 1: the same with
// x86_AVX512_STATE64; 2: set B with the AVX flavor, then C with the AVX-512
// flavor (got, changed, set). Prints one field per case; returns 1 when
// anything differs. refuse: the AVX-512 flavor must be refused (KERN_FAILURE),
// as the route does for another thread where it copies at CPUID's offsets;
// otherwise a refusal is recorded, not a failure.
static int ts_case(int c, int refuse, uint64_t *seed, char *line, size_t cap) {
	struct ts_arg *t = calloc(1, sizeof *t);
	struct pat want;
	fill(&t->in, seed);
	want = t->in;
	pthread_t th;
	pthread_create(&th, NULL, ts_main, t);
	while (t->flag != 1) {
	}
	mach_port_t port = pthread_mach_thread_np(th);
	kern_return_t ks = thread_suspend(port), kg = KERN_SUCCESS, kset = KERN_SUCCESS;
	kern_return_t kg2 = KERN_SUCCESS;
	int dget = 0, dget2 = 0, fail;
	char b1[64], b2[64], b3[64];
	x86_avx_state64_t a;
	x86_avx512_state64_t z;
	mach_msg_type_number_t n;
	if (c == 0) {
		n = x86_AVX_STATE64_COUNT;
		kg = thread_get_state(port, x86_AVX_STATE64, (thread_state_t)&a, &n);
		dget = kg ? -1 : fs_differ(&a, &want);
		kset = thread_set_state(port, x86_AVX_STATE64, (thread_state_t)&a, x86_AVX_STATE64_COUNT);
	} else if (c == 1) {
		n = x86_AVX512_STATE64_COUNT;
		kg = thread_get_state(port, x86_AVX512_STATE64, (thread_state_t)&z, &n);
		dget = kg ? -1 : fs_differ(&z, &want);
		if (!kg)
			kset = thread_set_state(port, x86_AVX512_STATE64, (thread_state_t)&z,
			                        x86_AVX512_STATE64_COUNT);
	} else {
		struct pat b, cc;
		fill(&b, seed);
		fill(&cc, seed);
		n = x86_AVX_STATE64_COUNT;
		kg = thread_get_state(port, x86_AVX_STATE64, (thread_state_t)&a, &n);
		fs_put(&a, &b);
		kset = thread_set_state(port, x86_AVX_STATE64, (thread_state_t)&a, x86_AVX_STATE64_COUNT);
		n = x86_AVX512_STATE64_COUNT;
		kg2 = thread_get_state(port, x86_AVX512_STATE64, (thread_state_t)&z, &n);
		dget = kg2 ? -1 : fs_differ(&z, &b); // the AVX-512 get sees what the AVX set wrote
		if (!kg2) {
			fs_put(&z, &cc);
			kern_return_t k = thread_set_state(port, x86_AVX512_STATE64, (thread_state_t)&z,
			                                   x86_AVX512_STATE64_COUNT);
			if (k)
				kset = k;
		}
		n = x86_AVX_STATE64_COUNT;
		kern_return_t k = thread_get_state(port, x86_AVX_STATE64, (thread_state_t)&a, &n);
		dget2 = k ? -1 : fs_differ(&a, kg2 ? &b : &cc);
		want = kg2 ? b : cc;
	}
	kern_return_t kr = thread_resume(port);
	t->flag = 2;
	pthread_join(th, NULL);
	int dout = differ(&want, &t->out);
	// The AVX-512 flavor's get: refused where it must be, and otherwise either
	// refused (recorded) or right. What the thread holds after a set, and the
	// AVX flavor's answers, are always checked.
	kern_return_t k512 = c == 1 ? kg : c == 2 ? kg2 : KERN_SUCCESS;
	int refused = c != 0 && k512 != KERN_SUCCESS;
	int refusal_wrong = c != 0 && refuse && k512 != KERN_FAILURE;
	fail = ks || kr || (c != 1 && kg) || kset || dout || dget > 0 || dget2 > 0 ||
	       (dget < 0 && !refused) || refusal_wrong;
	static const char *const names[3] = {"avx-unchanged", "avx512-unchanged",
	                                     "avx-then-avx512-changed"};
	snprintf(line, cap, " %s: kr suspend %d get %d set %d get2 %d resume %d, get %s%s%s, thread %s%s;",
	         names[c], ks, kg, kset, kg2, kr, dget < 0 ? "refused" : bits_str(dget, b1),
	         c == 2 ? ", get after set " : "",
	         c == 2 ? (dget2 < 0 ? "refused" : bits_str(dget2, b2)) : "", bits_str(dout, b3),
	         refusal_wrong ? ", AVX-512 flavor not refused" : "");
	free(t);
	return fail;
}

// --- (i) under load ---------------------------------------------------------
static atomic_int stop, ready;
struct worker {
	pthread_t th;
	int idx;
	uint64_t xcr0;
	int bits, fbits;
	unsigned long bad;
};
static uint64_t xgetbv0(void) {
	uint32_t lo, hi;
	__asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
	return ((uint64_t)hi << 32) | lo;
}
static void *worker_main(void *a) {
	struct worker *w = a;
	uint64_t seed = 0x9e3779b97f4a7c15ull * (uint64_t)(w->idx + 1);
	unsigned long r = 0;
	// The thread's TLS before any signal reaches it (a first TLS access in a
	// handler would allocate).
	sp_cur = NULL;
	sp_frame_bits = 0;
	atomic_fetch_add(&ready, 1);
	while (!atomic_load_explicit(&stop, memory_order_relaxed)) {
		long spin = (r & 1) ? 20000 + (long)(xs(&seed) & 0xffff) : 0;
		w->bad += (unsigned long)one_round(&seed, spin, &w->bits);
		if (r == 0)
			w->xcr0 = xgetbv0();
		r++;
	}
	w->fbits = sp_frame_bits;
	atomic_fetch_add(&C.rounds, r);
	atomic_fetch_add(&C.bad, w->bad);
	return NULL;
}
static struct worker *W;
static int NW;
static atomic_ulong sent;
static void *sender_main(void *a) {
	(void)a;
	while (!atomic_load_explicit(&stop, memory_order_relaxed)) {
		for (int i = 0; i < NW; i++)
			if (pthread_kill(W[i].th, SIGUSR1) == 0)
				atomic_fetch_add_explicit(&sent, 1, memory_order_relaxed);
		usleep(200);
	}
	return NULL;
}
static atomic_ulong churned, churn_bad;
static void *churn_one(void *a) {
	uint64_t seed = (uint64_t)(uintptr_t)a * 0x2545f4914f6cdd1dull + 1;
	int bits = 0;
	atomic_fetch_add(&churn_bad, (unsigned long)one_round(&seed, 0, &bits));
	atomic_fetch_add(&churn_bad, (unsigned long)one_round(&seed, 5000, &bits));
	return NULL;
}
static void *churn_main(void *a) {
	(void)a;
	uintptr_t i = 1;
	while (!atomic_load_explicit(&stop, memory_order_relaxed)) {
		pthread_t th[8];
		int n = 0;
		for (; n < 8; n++)
			if (pthread_create(&th[n], NULL, churn_one, (void *)i++))
				break;
		for (int k = 0; k < n; k++)
			pthread_join(th[k], NULL);
		atomic_fetch_add(&churned, (unsigned long)n);
	}
	return NULL;
}

// --- (ii) the thread's own state --------------------------------------------
static mach_port_t self_port;
static x86_avx512_state64_t self_z;
static kern_return_t self_kg, self_ks;
static struct evex self_new; // what the set puts in
static void self_get(void) {
	mach_msg_type_number_t n = x86_AVX512_STATE64_COUNT;
	self_kg = thread_get_state(self_port, x86_AVX512_STATE64, (thread_state_t)&self_z, &n);
}
static void self_set(void) {
	self_get();
	if (self_kg)
		return;
	memcpy(&self_z.__fpu_zmm16, self_new.zmm16, 64);
	memcpy(&self_z.__fpu_k3, &self_new.k3, 8);
	self_ks = thread_set_state(self_port, x86_AVX512_STATE64, (thread_state_t)&self_z,
	                           x86_AVX512_STATE64_COUNT);
}
// Returns 1 when anything is wrong; prints the ii-self line.
static int self_case(uint64_t *seed) {
	struct evex in, out;
	self_port = mach_thread_self();
	// Bind the calls before the case (a first call's binding runs dyld's code).
	self_get();
	fill_evex(&in, seed);
	memset(&out, 0, sizeof out);
	self_kg = self_ks = KERN_SUCCESS;
	sp_self_round(&in, &out, self_get);
	kern_return_t gk = self_kg;
	int g16 = !gk && !memcmp(&self_z.__fpu_zmm16, in.zmm16, 64);
	int gk3 = !gk && !memcmp(&self_z.__fpu_k3, &in.k3, 8);
	int kept = !memcmp(out.zmm16, in.zmm16, 64) && out.k3 == in.k3;
	fill_evex(&in, seed);
	fill_evex(&self_new, seed);
	memset(&out, 0, sizeof out);
	self_kg = self_ks = KERN_SUCCESS;
	sp_self_round(&in, &out, self_set);
	kern_return_t sk = self_kg, ss = self_ks;
	int s16 = !sk && !ss && !memcmp(out.zmm16, self_new.zmm16, 64);
	int sk3 = !sk && !ss && out.k3 == self_new.k3;
	mach_port_deallocate(mach_task_self(), self_port);
	printf("ii-self get kr %d: zmm16 %s k3 %s, registers %s; set kr %d/%d: zmm16 %s k3 %s\n", gk,
	       g16 ? "right" : "wrong", gk3 ? "right" : "wrong", kept ? "kept" : "lost", sk, ss,
	       s16 ? "right" : "wrong", sk3 ? "right" : "wrong");
	return !(g16 && gk3 && kept && s16 && sk3);
}

int main(int argc, char **argv) {
	setvbuf(stdout, NULL, _IOLBF, 0);
	int secs = argc > 1 ? atoi(argv[1]) : 60;
	int ncpu = 1;
	size_t sz = sizeof ncpu;
	sysctlbyname("hw.ncpu", &ncpu, &sz, NULL, 0);
	NW = argc > 2 ? atoi(argv[2]) : 2 * ncpu;

	struct sigaction sa;
	memset(&sa, 0, sizeof sa);
	sa.sa_sigaction = on_signal;
	sa.sa_flags = SA_SIGINFO | SA_RESTART;
	sigemptyset(&sa.sa_mask);
	sigaction(SIGILL, &sa, NULL);
	sigaction(SIGUSR1, &sa, NULL);

	// env, after this thread's first FP use (a pattern round with no signal: a
	// spin of one pass)
	uint64_t seed = 0x1234567887654321ull;
	int bits0 = 0;
	one_round(&seed, 1, &bits0);
	uint32_t r7[4], rd[4], c5[4], c6[4], c7[4];
	__asm__ volatile("cpuid" : "=a"(r7[0]), "=b"(r7[1]), "=c"(r7[2]), "=d"(r7[3]) : "a"(7), "c"(0));
	__asm__ volatile("cpuid" : "=a"(rd[0]), "=b"(rd[1]), "=c"(rd[2]), "=d"(rd[3]) : "a"(0xd), "c"(0));
	__asm__ volatile("cpuid" : "=a"(c5[0]), "=b"(c5[1]), "=c"(c5[2]), "=d"(c5[3]) : "a"(0xd), "c"(5));
	__asm__ volatile("cpuid" : "=a"(c6[0]), "=b"(c6[1]), "=c"(c6[2]), "=d"(c6[3]) : "a"(0xd), "c"(6));
	__asm__ volatile("cpuid" : "=a"(c7[0]), "=b"(c7[1]), "=c"(c7[2]), "=d"(c7[3]) : "a"(0xd), "c"(7));
	int avx512f = -1;
	sz = sizeof avx512f;
	if (sysctlbyname("hw.optional.avx512f", &avx512f, &sz, NULL, 0) != 0)
		avx512f = -1;
	char prod[64] = "?", build[64] = "?";
	sz = sizeof prod;
	sysctlbyname("kern.osproductversion", prod, &sz, NULL, 0);
	sz = sizeof build;
	sysctlbyname("kern.osversion", build, &sz, NULL, 0);
	uint64_t xcr0 = xgetbv0();
	// The route promoted this thread at its first FP use: XCR0 has components
	// 5-7 while leaf 7 says no avx512f. Where it did, and the offsets are not
	// Intel's (1088, 1152, 1664), the route copies at CPUID's offsets and
	// refuses another thread's AVX-512 state.
	int routed = (xcr0 & 0xe0) == 0xe0 && !((r7[1] >> 16) & 1);
	int intel = c5[1] == 1088 && c6[1] == 1152 && c7[1] == 1664;
	int refuse = routed && !intel;
	printf("env xcr0 0x%llx leaf7_avx512f %u leafd_eax 0x%x hw.optional.avx512f %d offsets k %u "
	       "zmm_hi256 %u hi16_zmm %u (%s) route %s macos %s (%s) ncpu %d\n",
	       (unsigned long long)xcr0, (r7[1] >> 16) & 1, rd[0], avx512f, c5[1], c6[1], c7[1],
	       intel ? "intel" : "not intel", routed ? "on" : "off", prod, build, ncpu);
	int fail = bits0 != 0;
	char b1[64], b2[64];

	// (i) signals, one thread: 1000 rounds of ud2
	int sbits = 0, sbad = 0;
	for (int i = 0; i < 1000; i++)
		sbad += one_round(&seed, 0, &sbits);
	printf("i-signal rounds 1000 bad %d registers %s frame checks %lu bad %lu (%s) flavors", sbad,
	       bits_str(sbits, b1), atomic_load(&C.frame_checks), atomic_load(&C.frame_bad),
	       bits_str(sp_frame_bits, b2));
	for (int k = 0; k < MC_N; k++)
		if (atomic_load(&C.mc[k]))
			printf(" %s %lu", mc_name[k], atomic_load(&C.mc[k]));
	printf("\n");
	fail |= sbad || atomic_load(&C.frame_bad) || atomic_load(&C.mc[MC_OTHER]);

	// (i) thread state
	char line[1024] = "i-thread-state", part[320];
	for (int c = 0; c < 3; c++) {
		fail |= ts_case(c, refuse, &seed, part, sizeof part);
		strncat(line, part, sizeof line - strlen(line) - 1);
	}
	printf("%s\n", line);

	// (i) under load
	memset(&C, 0, sizeof C);
	sp_frame_bits = 0;
	W = calloc((size_t)NW, sizeof *W);
	uint64_t t0 = mach_absolute_time();
	for (int i = 0; i < NW; i++) {
		W[i].idx = i;
		pthread_create(&W[i].th, NULL, worker_main, &W[i]);
	}
	while (atomic_load(&ready) != NW)
		usleep(1000);
	pthread_t snd, chn;
	pthread_create(&snd, NULL, sender_main, NULL);
	pthread_create(&chn, NULL, churn_main, NULL);
	sleep((unsigned)secs);
	atomic_store(&stop, 1);
	pthread_join(snd, NULL);
	pthread_join(chn, NULL);
	int lbits = 0, fbits = 0, x_e7 = 0;
	uint64_t xother = 0;
	for (int i = 0; i < NW; i++) {
		pthread_join(W[i].th, NULL);
		lbits |= W[i].bits;
		fbits |= W[i].fbits;
		if (W[i].xcr0 == 0xe7)
			x_e7++;
		else
			xother = W[i].xcr0;
	}
	mach_timebase_info_data_t tb;
	mach_timebase_info(&tb);
	double el = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e9;
	printf("i-load secs %.1f threads %d rounds %lu bad %lu registers %s sync %lu async-sent %lu "
	       "async-taken %lu frame checks %lu bad %lu (%s) churned threads %lu bad %lu xcr0 0xe7 in "
	       "%d/%d",
	       el, NW, atomic_load(&C.rounds), atomic_load(&C.bad), bits_str(lbits, b1),
	       atomic_load(&C.sync), atomic_load(&sent), atomic_load(&C.async),
	       atomic_load(&C.frame_checks), atomic_load(&C.frame_bad), bits_str(fbits, b2),
	       atomic_load(&churned), atomic_load(&churn_bad), x_e7, NW);
	if (x_e7 != NW)
		printf(" (other 0x%llx)", (unsigned long long)xother);
	printf(" flavors");
	for (int k = 0; k < MC_N; k++)
		if (atomic_load(&C.mc[k]))
			printf(" %s %lu", mc_name[k], atomic_load(&C.mc[k]));
	printf("\n");
	fail |= atomic_load(&C.bad) || atomic_load(&C.frame_bad) || atomic_load(&churn_bad) ||
	        atomic_load(&C.mc[MC_OTHER]) || atomic_load(&C.frame_checks) == 0;

	// (ii) an EVEX thread's zmm0 upper half, zmm16 and k3 across a handler
	// that leaves them, then one that overwrites them: sigreturn puts back
	// what the frame holds, so both must come back, and the frame must be the
	// AVX-512 one and hold them.
	int evex_runs = 1;
	for (int w = 0; w < 2; w++) {
		struct evex in, out;
		fill_evex(&in, &seed);
		memset(&out, 0, sizeof out);
		ev_cur = &in;
		ev_write = w;
		ev_ud = 0;
		ev_frame_kind = -1;
		ev_frame_zmmh0 = ev_frame_zmm16 = ev_frame_k3 = -1;
		sp_evex_round(&in, &out);
		static const char *const fr[3] = {"not in it", "wrong", "right"};
		if (ev_ud) {
			evex_runs = 0;
			printf("ii handler-%s: EVEX refused (#UD)\n", w ? "writes" : "leaves");
			continue;
		}
		int zk = !memcmp(in.zmm0 + 32, out.zmm0 + 32, 32), sk = !memcmp(in.zmm16, out.zmm16, 64);
		int kk = in.k3 == out.k3;
		printf("ii handler-%s: zmm0-upper %s zmm16 %s k3 %s; frame %s: zmm0-upper %s zmm16 %s k3 %s\n",
		       w ? "writes" : "leaves", zk ? "kept" : "lost", sk ? "kept" : "lost",
		       kk ? "kept" : "lost", ev_frame_kind >= 0 ? mc_name[ev_frame_kind] : "?",
		       fr[ev_frame_zmmh0 + 1], fr[ev_frame_zmm16 + 1], fr[ev_frame_k3 + 1]);
		fail |= !(zk && sk && kk && ev_frame_zmmh0 == 1 && ev_frame_zmm16 == 1 && ev_frame_k3 == 1);
	}
	if (evex_runs)
		fail |= self_case(&seed);
	else
		printf("ii-self: EVEX refused (#UD)\n");
	printf("result %s\n", fail ? "FAIL" : "PASS");
	return fail ? 1 : 0;
}
