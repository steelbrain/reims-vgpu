// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Finding the images Graft binds, per architecture. On x86-64 the consumer and
// its providers register as kexts and land in a kernel collection, found through
// OSKext and the collection's fileset, and the bind retries on each catalogue
// send. On arm64 the boot firmware places them before Graft's kmod start and
// names each in the device tree, so the bind runs once with no retry and no
// OSKext. The binding itself (KernelBinding.cpp) is architecture-neutral and
// reaches both through this interface.
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
// placed before Graft starts, and returns true. False refuses the binding. The
// dormant consumer is found by placedBase, below.
bool installBindingTrigger(const RuntimeImage &kernel, void (*onBind)());

// Locate provider `identifier` and fill base (0 on a miss). x86-64 requires it
// registered and started through OSKext, finds its live header in the kernel
// collection, and retains it in hold for the caller to release; arm64 reads the
// Boot collection's fileset entry, where every provider is already present, and
// retains nothing. A miss that may resolve on a later send sets waiting; reason
// explains it.
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
// `identifier`, 0 on a miss: the binding finds the dormant consumer by it, and the
// arm64e route the owned client. x86-64 reads the Boot collection's fileset
// entry, where OpenCore injected the kext. arm64 reads its per-kext base from the
// boot firmware's device tree (/chosen/ambrosia/<identifier>), since the kexts
// Graft serves are placed in the Auxiliary collection, not the Boot one, and so
// are not Boot-collection fileset entries.
uint64_t placedBase(const mach_header_64 *bootCollection, const char *identifier);
} // namespace graft
