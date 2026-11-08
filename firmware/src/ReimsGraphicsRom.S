/** @file
  The reims-graphics device's option ROM: OptionRom.S's layout around its
  driver's image.

  Its PCI identity (ROM_VENDOR_ID, ROM_DEVICE_ID and ROM_CLASS_CODE) comes from
  the host wire header. That header is generated C and cannot be assembled, so
  the Makefile reads the three values from it with the preprocessor and passes
  them in.
**/

#include "OptionRom.S"
