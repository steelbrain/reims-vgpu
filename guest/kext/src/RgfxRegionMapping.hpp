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

// Before a store to the device's registers (a doorbell, a status): orders every store before it,
// the region's among them, before the register's store for the device. On arm64 the region's
// release stores and fences order them for the guest's CPUs alone (inner shareable), and the
// device is no CPU of the guest's, so the register's store takes a DMB OSHST first, as Linux's
// writel does. x86-64 keeps every store in order, and takes nothing.
inline void rgfxRegisterWriteBarrier()
{
#if RGFX_GUEST_ARM64
	__builtin_arm_dmb(2); // DMB OSHST
#endif
}

// A 32-bit load or store of the device's registers (BAR 0), each one instruction. On arm64 a
// hypervisor that emulates the registers decodes a guest's access to them from its abort's
// syndrome, which the CPU gives only for a single load or store without writeback; a pre- or
// post-indexed, pair or SIMD&FP access aborts with no syndrome (ISV 0), and KVM hands that to
// QEMU, which decodes it only with its own instruction emulator. A volatile access fixes the
// access's size and count, not the instruction the compiler picks, so on arm64 each is written
// out: an LDR or STR of a W register at a base register's address, as Linux's readl and writel
// are. x86-64 has no such limit and keeps the volatile access. BAR 2, the region, is RAM to the
// hypervisor (QEMU's reims-graphics-pci maps it so), and its accesses do not reach it.
inline uint32_t rgfxRegisterRead32(const volatile uint8_t *at)
{
#if defined(__aarch64__)
	uint32_t value;
	__asm__ volatile("ldr %w0, [%1]" : "=r"(value) : "r"(at) : "memory");
	return value;
#else
	return *reinterpret_cast<const volatile uint32_t *>(at);
#endif
}

inline void rgfxRegisterWrite32(volatile uint8_t *at, uint32_t value)
{
#if defined(__aarch64__)
	__asm__ volatile("str %w0, [%1]" : : "rZ"(value), "r"(at) : "memory");
#else
	*reinterpret_cast<volatile uint32_t *>(at) = value;
#endif
}

#endif // RGFX_REGION_MAPPING_HPP_
