/** @file
  The reims-graphics device's option ROM: OptionRom.S's layout around its
  driver's image, with the device's PCI identity.
**/

#include "ReimsGraphics.h"

#define ROM_VENDOR_ID   RG_PCI_VENDOR_ID
#define ROM_DEVICE_ID   RG_PCI_DEVICE_ID
#define ROM_CLASS_CODE  RG_PCI_CLASS_CODE

#include "OptionRom.S"
