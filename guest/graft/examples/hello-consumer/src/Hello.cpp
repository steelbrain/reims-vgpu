// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// A minimal bound consumer, to read alongside docs/consumers.md and
// sdk/README.md. Its src/graft.json declares one provider library
// (com.apple.iokit.IOGraphicsFamily), one System-collection class it derives
// from (IOAccelerator), and its own class (GraftHelloConsumer). The build
// through sdk/consumer.mk turns those declarations into a __DATA,__graft
// manifest and leaves the kext dormant.
//
// Graft binds every placed kext that carries a manifest, whatever its bundle
// identifier, so this example has its own, ai.aneesiqbal.graft.HelloConsumer
// (Info.plist and the kmod declaration below).
//
// At boot Graft resolves this class's inherited vtable slots and its imported
// calls into IOGraphicsFamily by symbol name, checks IOAccelerator's live size
// and vtable against what this kext was compiled with, writes the manifest's
// sites in one transaction, and opens the activation gate so the kernel starts
// the kext. In ioreg (ioreg -l -d 1 | grep Graft) a reader then sees
// GraftState = active, GraftClient.ai.aneesiqbal.graft.HelloConsumer beginning
// "bound:", and this example's own GraftHelloConsumerState = passed.
#include "Hello.h"
#include <IOKit/IORegistryEntry.h>
#include <mach/mach_types.h>

OSDefineMetaClassAndStructors(GraftHelloConsumer, IOAccelerator);

IOService *GraftHelloConsumer::probe(IOService *, SInt32 *) {
    return nullptr;
}

// Two import sites that Graft binds into IOGraphicsFamily, one of each form the
// manifest carries. The volatile function pointer is a data site: an eight-byte
// pointer Graft writes. The direct call to releaseAccelID below is a branch
// site: a call instruction whose displacement Graft rewrites. Reaching
// kIOReturnSuccess through both shows that each form was bound to the live
// function, not left at its injected placeholder.
IOReturn (*volatile graftHelloCreate)(IOOptionBits, IOAccelID *) = &IOAccelerator::createAccelID;

extern "C" kern_return_t hello_start(kmod_info_t *, void *) {
    IOAccelID identifier = 0;
    IOReturn result = graftHelloCreate(0, &identifier);
    if (result == kIOReturnSuccess)
        result = IOAccelerator::releaseAccelID(0, identifier);
    auto root = IORegistryEntry::getRegistryRoot();
    if (root)
        root->setProperty("GraftHelloConsumerState",
                          result == kIOReturnSuccess ? "passed" : "refused: imported call");
    return KERN_SUCCESS;
}
extern "C" kern_return_t hello_stop(kmod_info_t *, void *) {
    return KERN_FAILURE;
}
extern "C" {
KMOD_EXPLICIT_DECL(ai.aneesiqbal.graft.HelloConsumer, "1.0.0", hello_start, hello_stop)
}
