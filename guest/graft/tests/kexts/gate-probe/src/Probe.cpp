// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#include "Probe.h"
#include <IOKit/IOLib.h>
#include <IOKit/IORegistryEntry.h>
#include <libkern/c++/OSString.h>
#include <mach/mach_types.h>

OSDefineMetaClassAndStructors(GraftGateProbe, IOAccelerator);

// An imported System-KC call. Graft binds this site before it opens the gate, so
// a started personality calling it proves the bind landed, not just that the
// personality matched. The volatile pointer keeps the import site.
IOReturn (*volatile graft_gate_create)(IOOptionBits, IOAccelID *) = &IOAccelerator::createAccelID;

// The category this personality carries, for the serial log and the ioreg record.
static const char *probeCategory(IOService *self) {
    auto category = OSDynamicCast(OSString, self->getProperty("GraftGateCategory"));
    return category ? category->getCStringNoCopy() : "?";
}

// Accept the match so start() runs. With the passive IOPropertyMatch gate the
// probe is not even constructed until Graft sets the gate property on the nub,
// so reaching probe() at all means the gate deferred matching until the bind;
// the serial line times it against Graft's property-set line.
IOService *GraftGateProbe::probe(IOService *, SInt32 *) {
    IOLog("reims-graft-gate: probe %s\n", probeCategory(this));
    return this;
}

bool GraftGateProbe::start(IOService *) {
    // The personality the gate deferred now starts. Exercise the bound import to
    // confirm the bind, then record the outcome on the registry root keyed by
    // category, so one property opening every category at once is visible in
    // ioreg. Decline to attach (return false): the test needs only that start()
    // was reached with the import bound, not a live accelerator on the test GPU,
    // and IOAccelerator's own start is heavy; the recorded property is the signal.
    const char *category = probeCategory(this);
    IOLog("reims-graft-gate: start %s\n", category);
    IOAccelID identifier = 0;
    IOReturn result = graft_gate_create(0, &identifier);
    if (result == kIOReturnSuccess)
        result = IOAccelerator::releaseAccelID(0, identifier);
    auto root = IORegistryEntry::getRegistryRoot();
    if (root)
        root->setProperty(category,
                          result == kIOReturnSuccess ? "passed" : "refused: imported call");
    return false;
}

extern "C" kern_return_t probe_start(kmod_info_t *, void *) {
    // Runs when the kext loads, before Graft opens the gate. Mark that the kext
    // loaded, so the forced-refuse arm (gate never opened) reads as loaded with
    // no category started, distinct from the kext being absent.
    auto root = IORegistryEntry::getRegistryRoot();
    if (root)
        root->setProperty("GraftGateProbeLoaded", "yes");
    return KERN_SUCCESS;
}
extern "C" kern_return_t probe_stop(kmod_info_t *, void *) {
    return KERN_FAILURE;
}
extern "C" {
KMOD_EXPLICIT_DECL(ai.aneesiqbal.ReimsGraphics, "1.0.0", probe_start, probe_stop)
}
