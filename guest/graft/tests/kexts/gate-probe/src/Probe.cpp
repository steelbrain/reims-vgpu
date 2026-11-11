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
    IOLog("graft-gate-probe: probe %s\n", probeCategory(this));
    return this;
}

bool GraftGateProbe::start(IOService *) {
    // The personality the gate deferred now starts. Exercise the bound import to
    // confirm the bind, then record the outcome on the registry root as
    // GraftGateProbe.<category>, so one property opening every category at once
    // is visible in ioreg. Decline to attach (return false): the test needs only that start()
    // was reached with the import bound, not a live accelerator on the host
    // bridge, and IOAccelerator's own start is heavy; the recorded property is
    // the signal.
    const char *category = probeCategory(this);
    IOLog("graft-gate-probe: start %s\n", category);
    IOAccelID identifier = 0;
    IOReturn result = graft_gate_create(0, &identifier);
    if (result == kIOReturnSuccess)
        result = IOAccelerator::releaseAccelID(0, identifier);
    char key[64];
    snprintf(key, sizeof(key), "GraftGateProbe.%s", category);
    auto root = IORegistryEntry::getRegistryRoot();
    if (root)
        root->setProperty(key, result == kIOReturnSuccess ? "passed" : "refused: imported call");
    return false;
}

extern "C" kern_return_t probe_start(kmod_info_t *, void *) {
    // The kernel starts the kext only once one of its personalities matches, so
    // this runs only after Graft opens the gate. GraftGateProbeLoaded therefore
    // marks a kext the gate let start: the forced-refuse build never starts, and
    // leaves the key absent.
    auto root = IORegistryEntry::getRegistryRoot();
    if (root)
        root->setProperty("GraftGateProbeLoaded", "yes");
    return KERN_SUCCESS;
}
extern "C" kern_return_t probe_stop(kmod_info_t *, void *) {
    return KERN_FAILURE;
}
extern "C" {
KMOD_EXPLICIT_DECL(ai.aneesiqbal.graft.GateProbe, "1.0.0", probe_start, probe_stop)
}
