// RgfxPat: the guest's page attribute table (IA32_PAT), whose entry 2 the x86-64 kext sets to
// write-back on every CPU (RgfxPat.cpp), and the record of every CPU's PAT that the accelerator's
// user client reads out (kRgfxAccelMethodCpuPats). Header-only and free of the kernel's headers
// but for the kext's own calls at its end, so the native test (tests/pat_test.cpp) and the guest's
// tool (guest/bundle/tools/pat_probe.c) build its values and its record as the kext does. C as in
// C++.
//
// Why entry 2. xnu's x86-64 pmap maps every page it does not manage (not its RAM: any PCI BAR)
// with PCD set and PWT and PAT clear, which is PAT entry 2, for every cache mode but
// write-combined and real-time (read from the public xnu source: pmap_get_cache_attributes,
// pmap_enter). Its PAT, 0x0001040600070406, holds UC- there. So the device's region, BAR2, which
// QEMU backs with host RAM, is uncached wherever the kext asks for write-back: its rings, the
// cursor, the device info, and every block the bundle maps write-back. An AMD host's nested
// paging and Windows' hypervisor (WHP, on Intel and AMD alike) both apply the guest's memory type
// there; KVM on an Intel host ignores it for RAM-backed pages. Measured on 26.7.2 guests, stores
// and loads through entry 2 ran at 78-202 MB/s on AMD KVM (a Radeon 780M host) and on WHP (the
// UHD 630 Mac mini), against 43-46 GB/s and 16.5/8.8 GB/s with entry 2 write-back; on Intel KVM
// at write-back speed either way.
//
// What entry 2 covers besides. Every other user of entry 2 in a guest reims-ctl starts is a
// device's registers that the host traps (BAR0, the doorbells of BAR4, the LAPIC, IOAPIC and HPET,
// the other emulated devices' BARs, OVMF's flash), where the memory type never reaches a device:
// each access exits and is emulated at its width, and the doorbell's store still follows the
// ring's stores in program order. RAM a driver asks uncached is coherent with QEMU's write-back
// view of it. xnu's physmap already maps every one of these addresses write-back (entry 0). A
// passed-through device's BAR would become write-back too; reims-ctl passes none through.
//
// Where it holds. Entry 6 stays write-combining, as xnu sets it. xnu's pat_init, at each CPU's
// start and at wake, writes the PAT only where entry 6 is not write-combining, and keeps a PAT
// whose entry 6 is (osfmk/i386/mtrr.c). A CPU taken offline and brought back online keeps its PAT
// under KVM and WHP (INIT keeps it), and xnu writes none then (measured). So the rewrite holds
// from the kext's start, and only a wake from S3, which resets every CPU's PAT before xnu writes
// its own, needs it again: reims-ctl starts its VMs with S3 off (ICH9-LPC.disable_s3=1), and its
// sleep is a dark wake, through which the measured guests wrote no PAT and every CPU kept the
// rewrite. A CPU offline when the kext starts is not reached; none is, at boot.
//
// Our own code.
#ifndef RGFX_PAT_HPP_
#define RGFX_PAT_HPP_

#include <stdint.h>

#include "RgfxUserClientABI.hpp"

#ifndef RGFX_GUEST_ARM64
#define RGFX_GUEST_ARM64 0
#endif

// The PAT's MSR, its entries' width, and the memory types its entries take (the Intel SDM, Vol.
// 3A, "Page Attribute Table"; AMD's APM, Vol. 2, "Page-Attribute Table Mechanism").
#define RGFX_MSR_PAT 0x277u
#define RGFX_PAT_ENTRY_BITS 8u
#define RGFX_PAT_UNCACHEABLE 0x00u
#define RGFX_PAT_WRITE_COMBINING 0x01u
#define RGFX_PAT_WRITE_BACK 0x06u
#define RGFX_PAT_UNCACHED_MINUS 0x07u
// The entry xnu's pmap gives a page it does not manage (PCD=1 PWT=0 PAT=0), which the kext sets to
// write-back, and the one it gives such a page asked write-combined (PCD=1 PWT=0 PAT=1).
#define RGFX_PAT_DEVICE_ENTRY 2u
#define RGFX_PAT_WRITE_COMBINED_ENTRY 6u
// xnu's PAT, as its pat_init leaves it on every CPU.
#define RGFX_PAT_XNU 0x0001040600070406ull
// A CPU number no CPU answered for, in the record: a PAT's reserved bits are zero, so no PAT is
// this.
#define RGFX_PAT_NONE 0xffffffffffffffffull

// The type entry `entry` (0 to 7) of `pat` holds.
static inline uint32_t rgfxPatEntry(uint64_t pat, uint32_t entry)
{
	return (uint32_t)((pat >> (entry * RGFX_PAT_ENTRY_BITS)) & 0xffu);
}

// `pat` with entry 2 write-back and every other entry as it is.
static inline uint64_t rgfxPatWithDeviceWriteBack(uint64_t pat)
{
	const uint32_t shift = RGFX_PAT_DEVICE_ENTRY * RGFX_PAT_ENTRY_BITS;
	return (pat & ~(0xffull << shift)) | ((uint64_t)RGFX_PAT_WRITE_BACK << shift);
}

// Whether a CPU's PAT is the one the kext keeps: entry 2 write-back, and entry 6
// write-combining, as xnu's pat_init sets it on every CPU it starts.
static inline int rgfxPatHeld(uint64_t pat)
{
	return pat != RGFX_PAT_NONE &&
	       rgfxPatEntry(pat, RGFX_PAT_DEVICE_ENTRY) == RGFX_PAT_WRITE_BACK &&
	       rgfxPatEntry(pat, RGFX_PAT_WRITE_COMBINED_ENTRY) == RGFX_PAT_WRITE_COMBINING;
}

// Every CPU's PAT, recorded by each CPU itself in a rendezvous: `pats` holds `count` entries, by
// CPU number, sized to the kernel's CPU count; `answered` counts the CPUs that recorded, and
// `outside` those whose number lies past `count`, which are counted and not kept. Each CPU writes
// only its own entry, and both counts atomically.
typedef struct RgfxCpuPats {
	uint64_t *pats;
	uint32_t count;
	uint32_t answered;
	uint32_t outside;
} RgfxCpuPats;

// Empties the record: every entry RGFX_PAT_NONE, both counts 0.
static inline void rgfxCpuPatsClear(RgfxCpuPats *t)
{
	for (uint32_t i = 0; i < t->count; i++)
		t->pats[i] = RGFX_PAT_NONE;
	t->answered = 0;
	t->outside = 0;
}

// CPU `cpu` records its PAT.
static inline void rgfxCpuPatsRecord(RgfxCpuPats *t, int cpu, uint64_t pat)
{
	__atomic_fetch_add(&t->answered, 1u, __ATOMIC_RELAXED);
	if (cpu < 0 || (uint32_t)cpu >= t->count) {
		__atomic_fetch_add(&t->outside, 1u, __ATOMIC_RELAXED);
		return;
	}
	t->pats[cpu] = pat;
}

#ifdef __cplusplus
// The kext's CpuPats feature bit (kRgfxKextFeatureCpuPats), where the kext reads the PAT: on
// x86-64. An arm64 CPU has no PAT.
constexpr uint64_t kRgfxKextFeatureCpuPatsRead = RGFX_GUEST_ARM64 ? 0 : kRgfxKextFeatureCpuPats;
#endif

#if defined(KERNEL) && defined(__x86_64__)
// The kext's calls (RgfxPat.cpp), x86-64 only.

// What one rewrite did: the CPUs that ran it, those whose PAT it changed, and its time.
typedef struct RgfxPatRun {
	uint32_t cpus;
	uint32_t changed;
	uint64_t ns;
} RgfxPatRun;

// Sets entry 2 of every online CPU's PAT to write-back where it is not already, each CPU with its
// interrupts off and through the SDM's sequence for a PAT change, and logs what it did, `why`
// naming the caller. Called with interrupts on and no spin lock held.
RgfxPatRun rgfxPatWriteBackEverywhere(const char *why);
// From the first call on, the rewrite runs again at each wake; rgfxPatStopWatching takes the
// interest back.
void rgfxPatWatchWakes(void);
void rgfxPatStopWatching(void);
// The kernel's CPU count, which sizes a record.
uint32_t rgfxPatCpuCount(void);
// Reads every online CPU's PAT into `t`, emptied first, each CPU its own with its interrupts off.
void rgfxPatRead(RgfxCpuPats *t);
#endif

#endif // RGFX_PAT_HPP_
