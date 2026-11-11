// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Finding the images Graft binds, per architecture. On x86-64 the consumers and
// their providers register as kexts and land in a kernel collection, found
// through OSKext and the collection's fileset, and the binds retry on each
// catalogue send. On arm64 the boot firmware places them before Graft's kmod
// start and names each in the device tree, so the binds run once with no retry.
// Both start each provider through OSKext before binding against it. The binding
// itself (KernelBinding.cpp) is architecture-neutral and reaches both through
// this interface.
#pragma once
#include "RuntimeImage.h"

class OSObject;

namespace graft {
// A placed kext located by bundle identifier.
struct Placement {
    uint64_t base = 0;        // its slid sub-header address; 0 when not available
    OSObject *hold = nullptr; // a reference that keeps it resident while bound (x86-64), else null
    bool waiting = false;     // true when base is 0 only because the kext may still appear
};

// Install the binding trigger from Graft's kmod start, after the write and route
// facilities. x86-64 routes the two catalogue entry points so each send calls
// onBind (bindConsumers) again; arm64 needs no trigger, since every image is
// placed before Graft starts, and returns true. False refuses the binding.
bool installBindingTrigger(const RuntimeImage &kernel, void (*onBind)());

// Call visit with each kext placed at boot, by bundle identifier and slid
// sub-header address: Graft's start keeps those that carry a manifest as its
// bound consumers. x86-64 walks the Boot collection's fileset entries, where
// OpenCore injects the kexts. arm64 reads each child of the boot firmware's
// /chosen/ambrosia node that has a base, since the firmware names every kext it
// places there by bundle identifier, in the Auxiliary collection rather than the
// Boot one. The identifier lasts only for the call.
void eachPlacedKext(const mach_header_64 *bootCollection,
                    void (*visit)(void *context, const char *identifier, uint64_t base),
                    void *context);

// Locate provider `identifier`, start it, and fill base (0 on a miss). x86-64
// requires it registered and started through OSKext, finds its live header in the
// kernel collection, and retains it in hold for the caller to release; arm64 reads
// the Boot collection's fileset entry, starts it through OSKext, and retains
// nothing. A miss that may resolve on a later send sets waiting; reason explains
// it.
Placement locate(const mach_header_64 *collection, const char *identifier, char *reason,
                 size_t reasonSize);

// Whether placed kext `identifier` has already started. Binding must happen while
// the consumer is dormant, so a started consumer is refused.
bool kextStarted(const char *identifier);

// Whether a write of `size` bytes at `address` in the placed consumer image is
// allowed to land there, per architecture. x86-64 binds branch sites in __TEXT,
// so any mapped, readable target is allowed. arm64e requires a writable data
// segment the kernel has not locked read-only, since its __DATA_CONST is mapped
// rw- plus SG_READ_ONLY and a store to it faults once the kernel locks it.
bool writableConsumerTarget(const RuntimeImage &consumer, uint64_t address, uint64_t size);

// The slid sub-header address of the kext placed with bundle identifier
// `identifier`, 0 on a miss. x86-64 reads the Boot collection's fileset entry,
// where OpenCore injected the kext; arm64e reads its per-kext base from the boot
// firmware's device tree (/chosen/ambrosia/<identifier>), since the kexts Graft
// serves land in the Auxiliary collection, not the Boot one. The identity gate
// (src/Start.cpp) accepts a client by it, and the arm64e route finds a client's
// image with it.
uint64_t placedBase(const mach_header_64 *bootCollection, const char *identifier);
} // namespace graft
