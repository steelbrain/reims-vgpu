// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#pragma once
#include <IOKit/graphics/IOAccelerator.h>

// Several gated personalities, one per IOMatchCategory, all naming this class on
// the test GPU. Each defers on the manifest gate until Graft binds the consumer
// and publishes the key; then every category's start() runs. The class binds a
// real System-KC superclass, so a started personality also proves the bind.
class GraftGateProbe : public IOAccelerator {
    OSDeclareDefaultStructors(GraftGateProbe);

public:
    IOService *probe(IOService *, SInt32 *) override;
    bool start(IOService *) override;
};
