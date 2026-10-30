/** @file
  PCI option ROM holding the driver's EFI image (PCI Firmware Specification
  3.0, UEFI specification 14.4.2): the EFI expansion ROM header, the PCI data
  structure, the image itself, and 0xFF padding to a multiple of 512 bytes.

  The Makefile assembles this with EFI_IMAGE set to the linked image's path
  and flattens the result to raw bytes. It is the apple-gfx-pci device's ROM
  unless an including file defines ROM_VENDOR_ID, ROM_DEVICE_ID and
  ROM_CLASS_CODE for another device first.
**/

#ifndef ROM_VENDOR_ID
#include "ReimsVgpu.h"
#define ROM_VENDOR_ID   REIMS_VGPU_VENDOR_ID
#define ROM_DEVICE_ID   REIMS_VGPU_DEVICE_ID
#define ROM_CLASS_CODE  REIMS_VGPU_CLASS_CODE
#endif

#define ROM_BLOCKS  ((RomEnd - RomStart) / 512)

  .section .rom, "a"

RomStart:
  .short  0xAA55                // Signature
  .short  ROM_BLOCKS            // InitializationSize, in 512-byte units
  .long   0x0EF1                // EfiSignature
  .short  11                    // EfiSubsystem: boot service driver, as linked
  .short  0x8664                // EfiMachineType: x64
  .short  0                     // CompressionType: none
  .zero   8                     // Reserved
  .short  EfiImage - RomStart   // EfiImageHeaderOffset
  .short  Pcir - RomStart       // PcirOffset

  .balign 4, 0
Pcir:
  .ascii  "PCIR"
  .short  ROM_VENDOR_ID
  .short  ROM_DEVICE_ID
  .short  0                     // DeviceListOffset
  .short  PcirEnd - Pcir        // Length
  .byte   3                     // Revision
  .byte   ROM_CLASS_CODE & 0xFF, (ROM_CLASS_CODE >> 8) & 0xFF, ROM_CLASS_CODE >> 16
  .short  ROM_BLOCKS            // ImageLength, in 512-byte units
  .short  0                     // CodeRevision
  .byte   3                     // CodeType: EFI image
  .byte   0x80                  // Indicator: last image in the ROM
  .short  0                     // MaxRuntimeImageLength
  .short  0                     // ConfigUtilityCodeHeaderOffset
  .short  0                     // DMTFCLPEntryPointOffset
PcirEnd:

EfiImage:
  .incbin EFI_IMAGE
  .balign 512, 0xFF
RomEnd:
