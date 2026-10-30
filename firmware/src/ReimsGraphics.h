/** @file
  Private data and interfaces of the reims-graphics device's UEFI driver: a
  Graphics Output Protocol on the device's framebuffer in its region (BAR 2),
  set up through the control ring (docs/graphics-kext/design.md, "Stage 2
  wire protocol").
**/

#ifndef REIMS_GRAPHICS_H_
#define REIMS_GRAPHICS_H_

#ifndef __ASSEMBLER__

#include "reims_graphics_wire.h"
#include "Uefi.h"

#define EFI_DEVICE_ERROR  EFIERR (7)
#define EFI_TIMEOUT       EFIERR (18)

typedef
VOID
(EFIAPI *EFI_STALL)(
  IN UINTN  Microseconds
  );

//
// The mode used when the device names no connected display with a mode.
//
#define REIMS_GRAPHICS_FALLBACK_WIDTH   1920
#define REIMS_GRAPHICS_FALLBACK_HEIGHT  1080

//
// A scanout row's bytes are a multiple of this, as the guest's display kext
// lays them out, so the boot frame keeps its layout when the kext takes the
// display over.
//
#define REIMS_GRAPHICS_STRIDE_ALIGN  256

//
// How long a command waits for its answer, in polls of the completion ring
// REIMS_GRAPHICS_POLL_US apart: 2 s.
//
#define REIMS_GRAPHICS_POLL_US     10
#define REIMS_GRAPHICS_POLL_LIMIT  200000

#define REIMS_GRAPHICS_PRIVATE_DATA_SIGNATURE  SIGNATURE_32 ('R', 'G', 'F', 'X')

typedef struct REIMS_GRAPHICS_PRIVATE_DATA  REIMS_GRAPHICS_PRIVATE_DATA;

struct REIMS_GRAPHICS_PRIVATE_DATA {
  UINTN                                   Signature;
  REIMS_GRAPHICS_PRIVATE_DATA             *Next;      // the driver's other controllers
  EFI_HANDLE                              Controller;
  EFI_HANDLE                              Handle;     // the Graphics Output child
  EFI_PCI_IO_PROTOCOL                     *PciIo;
  UINT64                                  OriginalPciAttributes;
  EFI_DEVICE_PATH_PROTOCOL                *GopDevicePath;
  EFI_GRAPHICS_OUTPUT_PROTOCOL            GraphicsOutput;
  EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE       GraphicsOutputMode;
  EFI_GRAPHICS_OUTPUT_MODE_INFORMATION    ModeInformation;

  //
  // The region: BAR 2's address, and the size the device gives.
  //
  EFI_PHYSICAL_ADDRESS                    RegionBase;
  UINT64                                  RegionSize;

  //
  // The control rings: the guest's producer count of the submission ring,
  // its consumer count of the completion ring, and the last command's
  // number.
  //
  UINT64                                  SubmitProducer;
  UINT64                                  EventConsumer;
  UINT64                                  Seqno;

  //
  // The one mode offered, on the display Display names.
  //
  UINT32                                  Display;
  UINT32                                  Width;
  UINT32                                  Height;
  UINT32                                  Stride;     // bytes
};

#define REIMS_GRAPHICS_PRIVATE_DATA_FROM_GRAPHICS_OUTPUT_THIS(a) \
  BASE_CR (a, REIMS_GRAPHICS_PRIVATE_DATA, GraphicsOutput)

extern EFI_HANDLE         gImageHandle;
extern EFI_BOOT_SERVICES  *gBS;

extern EFI_GUID  gEfiPciIoProtocolGuid;

extern EFI_DRIVER_BINDING_PROTOCOL   gReimsGraphicsDriverBinding;
extern EFI_COMPONENT_NAME_PROTOCOL   gReimsGraphicsComponentName;
extern EFI_COMPONENT_NAME2_PROTOCOL  gReimsGraphicsComponentName2;

/**
  Send SET_SCANOUT for the private data's mode at RGFX_ALLOCATIONS_OFFSET, or
  blank the display when Blank is set, and wait for its COMPLETE.
**/
EFI_STATUS
ReimsGraphicsSetScanout (
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private,
  IN BOOLEAN                      Blank
  );

VOID
ReimsGraphicsGraphicsOutputInit (
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private
  );

#endif // __ASSEMBLER__

#endif
