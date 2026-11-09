// RgfxSuperpageRecord: the guest's record of whether its kernel has a 2 MiB superpage left, which
// the kext keeps for every process of the guest (kRgfxAccelMethodSuperpages,
// kRgfxKextFeatureSuperpages): the bundle's slabs of a process's own pages ask the kernel for
// superpages only while it says so (guest/bundle/src/RgfxSlabs.hpp, RgfxPagesSlabSource).
//
// Why one record for every process: a superpage the kernel cannot give costs the asking thread
// the kernel's search for a contiguous 2 MiB (vm_map_enter's superpage path, in cpm_allocate,
// which sleeps as it goes and moves pages to make a run, vm_page_relocate). Measured with
// spindump on an 8 GiB x86-64 macOS 26.7.2 guest on the Radeon 780M host, its frames read
// against the 25G241 KDK's symbols: the guest had none left a minute after its boot, and each
// later ask took 140 to 190 ms before its refusal, in the first slab of every process (a video
// decoder session's first access unit, Safari's GPU process's layer drawing), which then took
// plain pages anyway. A 16 GiB macOS 26.6.1 guest on the RTX 5080 host gave every one asked for
// at once. A process's own memory of a refusal would spare only its later slabs; the record
// spares the first of every other process's too.
//
// A refusal closes the record; a slab of superpages given back to the kernel opens it, as a
// contiguous 2 MiB is free again. Free of the kernel's headers, so the native test
// (tests/user_client_words_test.cpp) builds it as the kext does.
//
// Our own code.
#ifndef RGFX_SUPERPAGE_RECORD_HPP_
#define RGFX_SUPERPAGE_RECORD_HPP_

#include "RgfxUserClientABI.hpp"

// Takes `note` (kRgfxSuperpages*) into the record `*none`, true where the kernel refused the last
// ask since a slab of superpages went back, and answers what kRgfxAccelMethodSuperpages does: 1
// where a process may ask the kernel for superpages, 0 where not, -1 for a note that is none of
// them, the record unchanged.
static inline int rgfxSuperpagesNote(bool *none, unsigned long long note)
{
	switch (note) {
	case kRgfxSuperpagesAsk:
		break;
	case kRgfxSuperpagesRefused:
		__atomic_store_n(none, true, __ATOMIC_RELAXED);
		break;
	case kRgfxSuperpagesGivenBack:
		__atomic_store_n(none, false, __ATOMIC_RELAXED);
		break;
	default:
		return -1;
	}
	return __atomic_load_n(none, __ATOMIC_RELAXED) ? 0 : 1;
}

#endif // RGFX_SUPERPAGE_RECORD_HPP_
