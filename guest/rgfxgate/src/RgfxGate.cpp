// IOResources matching loads this kext; initialization runs from its start glue.
#include <IOKit/IOService.h>
#include "RgfxGate.hpp"

class RgfxGate : public IOService {
    OSDeclareDefaultStructors(RgfxGate)
};
OSDefineMetaClassAndStructors(RgfxGate, IOService)

void graft_client_initialize() {
    if (!graft_client_ready(rgfxgate::kClientID))
        return;
    if (!rgfxgate::installHooks())
        return;
    rgfxgate::startInstaller();
}
