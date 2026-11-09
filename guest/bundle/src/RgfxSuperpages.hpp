/*
 * RgfxSuperpages: the process's own 2 MiB superpage (x86-64), one physically contiguous run of
 * guest RAM, zeroed and wired, as the pages slab source takes it (RgfxSlabs.hpp, RgfxPagesMemory),
 * one at a time on the slabs' thread.
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
 * call for one superpage takes down only what it made, in every process. So the call here asks
 * for one, anywhere, and a slab larger than a superpage is plain pages (RgfxPagesSlabSource).
 * tools/spguard shows the several-superpage call's exception in a guest.
 *
 * Our own code; it names only the SDK's public Mach VM calls.
 */

#ifndef RGFX_SUPERPAGES_HPP_
#define RGFX_SUPERPAGES_HPP_

#include <cstdint>

/// A superpage's bytes.
constexpr uint64_t kRgfxSuperpage = 2ull << 20;

/// One superpage anywhere in this task (mach_vm_allocate, VM_FLAGS_SUPERPAGE_SIZE_2MB), zeroed and
/// wired, tagged VM_MEMORY_IOACCELERATOR, so vmmap and footprint file it where Metal's own memory
/// shows: 0 (KERN_SUCCESS) and *bytes, or the kernel's refusal, with nothing left mapped
/// (KERN_NO_SPACE where no contiguous 2 MiB is free, KERN_RESOURCE_SHORTAGE past the user wire
/// budget). Its bytes go back with one mach_vm_deallocate of kRgfxSuperpage.
int rgfxTakeSuperpage(uint8_t **bytes);

#endif // RGFX_SUPERPAGES_HPP_
