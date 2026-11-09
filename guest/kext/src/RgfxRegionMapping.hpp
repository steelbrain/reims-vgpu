// RgfxRegionMapping: how a process's mapping of the device's region behaves in the guest kernel
// the kext is built for, which differs between the kext's two architectures. The kext's Makefile
// builds the arm64e kext with RGFX_GUEST_ARM64 set; the x86-64 kext, and the native tests, take
// x86-64's values. Header-only and free of the kernel's headers, so the native tests
// (tests/arm64_region_test.cpp among them) build it as the kext does.
//
// Our own code.
#ifndef RGFX_REGION_MAPPING_HPP_
#define RGFX_REGION_MAPPING_HPP_

#include <stdint.h>

#include "RgfxUserClientABI.hpp"

#ifndef RGFX_GUEST_ARM64
#define RGFX_GUEST_ARM64 0
#endif

#if RGFX_GUEST_ARM64
// The page an arm64 process's memory is mapped in: 16 KiB. A mapping made or taken back at a
// 4 KiB offset in such a process acts on the whole 16 KiB page around it.
constexpr uint64_t kRgfxProcessPage = 0x4000;
// Whether a process's write-combined mapping of the region is ordinary memory: not on arm64.
// xnu's arm64 pmap maps a write-combined page that is not the guest's RAM, as the region's are
// not, as Device-GRE memory (read from the public xnu source: wimg_to_pte, VM_WIMG_WCOMB where
// is_dram_addr is false), where an unaligned access or bzero's DC ZVA faults: the process dies
// of SIGBUS. Write-back maps the same page as Normal write-back memory, which takes both.
constexpr bool kRgfxRegionWriteCombines = false;
#else
// The page an x86-64 process's memory is mapped in: 4 KiB.
constexpr uint64_t kRgfxProcessPage = 0x1000;
// x86-64's pmap maps a write-combined page with the guest PAT's WC, as the kext asks.
constexpr bool kRgfxRegionWriteCombines = true;
#endif

// The kext's write-combined feature bit, where the region's write-combined mappings are ordinary
// memory: a bundle that does not find it in the kext's feature word maps every block write-back.
constexpr uint64_t kRgfxKextFeatureRegionWriteCombined =
    kRgfxRegionWriteCombines ? kRgfxKextFeatureWriteCombined : 0;

#endif // RGFX_REGION_MAPPING_HPP_
