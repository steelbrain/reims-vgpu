// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#pragma once
#include <IOKit/graphics/IOAccelerator.h>

// The smallest bound consumer: one class that derives from a System-collection
// class. IOAccelerator lives in IOGraphicsFamily, which the Boot collection
// does not hold, so OpenCore's linker cannot resolve this class's inherited
// slots and imports at inject time. Graft resolves them by symbol name at boot,
// once IOGraphicsFamily is registered (docs/design.md, "Binding a consumer").
// IOAccelerator is a convenient real superclass to show that: deriving from it
// needs the live class's size, vtable length and slots, which Graft checks.
//
// The personality (Info.plist) matches IOResources, the provider every guest of
// both architectures registers, so the example needs no particular device. It
// carries an IOMatchCategory of its own, so it does not compete with the other
// drivers that match IOResources. Graft opens the gate there as on any provider,
// by setting the gate property on IOResources and registering it again, which is
// what IOService::publishResource does. A real driver's personalities match its
// own device instead.
//
// probe() declines the match, so nothing attaches; the example only has to bind
// and start, not drive hardware.
class GraftHelloConsumer : public IOAccelerator {
    OSDeclareDefaultStructors(GraftHelloConsumer);

public:
    IOService *probe(IOService *, SInt32 *) override;
};
