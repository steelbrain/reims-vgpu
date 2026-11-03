/*
 * RgfxSuperpages: the process's own 2 MiB superpages (x86-64), each one physically contiguous run
 * of guest RAM, zeroed and wired, as the pages slab source takes them (RgfxSlabs.hpp,
 * RgfxPagesMemory).
 *
 * ONE SUPERPAGE A CALL. A kernel call for several superpages that it cannot fill (no free
 * contiguous 2 MiB for one of them) fails, and its cleanup takes down the whole range it was asked
 * for, past the superpages it made. The part it never made is unmapped, and taking down unmapped
 * memory is a deallocation gap, which the kernel answers with an EXC_GUARD exception
 * (GUARD_TYPE_VIRT_MEMORY, DEALLOC_GAP, at the end of the last superpage it made). Read from the
 * public xnu source, xnu-12377.101.15 and .121.6 (macOS 26.4 and 26.5): vm_map_enter's superpage
 * path shrinks `size` to what it made, then deletes [start, end). xnu-12377.81.4 and earlier
 * delete [start, start + size). The exception is fatal in every process whose VM guard is on:
 * Apple's platform binaries and the hardened browser processes (WebKit's GPU process among them)
 * have it, a third-party app does not (task_set_exc_guard_default), where the call just fails. A
 * call for one superpage takes down only what it made, in every process.
 *
 * So one superpage is one call, anywhere; and several are a range reserved first (anywhere, at a
 * 2 MiB boundary, no access), then for each 2 MiB of it in turn, that piece of the reservation
 * given back and a superpage made in the hole it leaves (fixed, not overwriting). Where the kernel
 * refuses one, what the call made was its own one entry, which it takes down whole, and the run
 * gives back the superpages before the hole and the reservation after it: two deallocations, each
 * of memory the run holds, so neither meets a gap, and the hole is left as it is (empty, or
 * another thread's, which took it in the moment it was free). Not an overwriting call over the
 * reservation: where one fails, vm_map_enter puts back what it took only where no entry starts
 * at the range's end on xnu-12377.81.4 and earlier (vm_map_lookup_entry(map, end) is true there),
 * and the reservation's next piece always does, so a refusal would leave a hole in the run.
 *
 * Our own code; it names only the SDK's public Mach VM calls.
 */

#ifndef RGFX_SUPERPAGES_HPP_
#define RGFX_SUPERPAGES_HPP_

#include <cstdint>

/// A superpage's bytes.
constexpr uint64_t kRgfxSuperpage = 2ull << 20;

/// The kernel's VM calls rgfxTakeSuperpages makes: the kernel's own (rgfxKernelSuperpageVm) or a
/// test's. Each returns 0 (KERN_SUCCESS) or the kernel's refusal.
struct RgfxSuperpageVm {
	void *context = nullptr;
	/// `size` bytes (a multiple of 2 MiB) of superpages, zeroed and wired: anywhere, *at set,
	/// where `fixed` is false; at *at, where nothing lies, where it is true.
	int (*superpages)(void *context, uint64_t *at, uint64_t size, bool fixed) = nullptr;
	/// `size` bytes anywhere at a 2 MiB boundary, with no access and no pages: the range several
	/// superpages are made in, piece by piece. *at set.
	int (*reserve)(void *context, uint64_t *at, uint64_t size) = nullptr;
	/// Gives back `size` bytes at `at`.
	void (*deallocate)(void *context, uint64_t at, uint64_t size) = nullptr;
};

/// The kernel's own calls on this task: mach_vm_allocate (VM_FLAGS_SUPERPAGE_SIZE_2MB, fixed or
/// anywhere), mach_vm_map (the reservation) and mach_vm_deallocate, every range tagged
/// VM_MEMORY_IOACCELERATOR, so vmmap and footprint file it where Metal's own memory shows.
RgfxSuperpageVm rgfxKernelSuperpageVm();

/// `size` bytes (a positive multiple of 2 MiB) of superpages, one kernel call each: 0 and *bytes,
/// or the first refusal with nothing of the call's left mapped (KERN_INVALID_ARGUMENT, uncalled,
/// for any other size).
int rgfxTakeSuperpages(const RgfxSuperpageVm &vm, uint64_t size, uint8_t **bytes);

/// Gives back what rgfxTakeSuperpages made: one deallocation of the whole range.
void rgfxGiveSuperpages(const RgfxSuperpageVm &vm, uint8_t *bytes, uint64_t size);

#endif // RGFX_SUPERPAGES_HPP_
