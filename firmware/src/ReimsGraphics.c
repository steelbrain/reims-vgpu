/** @file
  Entry point, driver binding and control ring of the reims-graphics device's
  UEFI driver.

  Start resets the device, adopts the control rings' counters, asks the
  device for its displays and offers one mode: the preferred mode of the
  first connected display, or 1920x1080 when there is none. The framebuffer
  is the region's memory from RGFX_ALLOCATIONS_OFFSET, and SetMode points the
  display at it with SET_SCANOUT. Firmware takes no interrupts: a command
  polls the completion ring for its answer.
**/

#include "ReimsGraphics.h"

EFI_HANDLE         gImageHandle;
EFI_BOOT_SERVICES  *gBS;

STATIC EFI_SYSTEM_TABLE  *mSystemTable;

STATIC EFI_DEVICE_PATH_UTILITIES_PROTOCOL  *mDevicePathUtilities;
STATIC REIMS_GRAPHICS_PRIVATE_DATA         *mControllers;

EFI_GUID  gEfiDevicePathProtocolGuid          = EFI_DEVICE_PATH_PROTOCOL_GUID;
EFI_GUID  gEfiDevicePathUtilitiesProtocolGuid = EFI_DEVICE_PATH_UTILITIES_PROTOCOL_GUID;
EFI_GUID  gEfiDriverBindingProtocolGuid       = EFI_DRIVER_BINDING_PROTOCOL_GUID;
EFI_GUID  gEfiComponentNameProtocolGuid       = EFI_COMPONENT_NAME_PROTOCOL_GUID;
EFI_GUID  gEfiComponentName2ProtocolGuid      = EFI_COMPONENT_NAME2_PROTOCOL_GUID;
EFI_GUID  gEfiPciIoProtocolGuid               = EFI_PCI_IO_PROTOCOL_GUID;
EFI_GUID  gEfiGraphicsOutputProtocolGuid      = EFI_GRAPHICS_OUTPUT_PROTOCOL_GUID;

//
// The control rings in the region, and their header fields.
//
#define RING_HEADER(Private, Offset) \
  ((volatile UINT8 *)(UINTN)((Private)->RegionBase + (Offset)))
#define SUBMIT_RING(Private)  RING_HEADER (Private, RGFX_CONTROL_SUBMIT_OFFSET)
#define EVENT_RING(Private)   RING_HEADER (Private, RGFX_CONTROL_COMPLETE_OFFSET)
#define RING_U64(Ring, Field)  ((volatile UINT64 *)((Ring) + (Field)))
#define RING_U32(Ring, Field)  ((volatile UINT32 *)((Ring) + (Field)))
#define RING_BYTES(Ring)       ((Ring) + RGFX_RING_HEADER_SIZE)

//
// The ring header's fields, by their offset in it, and the control ring's
// byte-ring size (a power of two, so an index wraps with a mask) and
// alignment.
//
#define RING_PRODUCER_OFFSET       __builtin_offsetof (RgfxRingHeader, producer)
#define RING_CONSUMER_OFFSET       __builtin_offsetof (RgfxRingHeader, consumer)
#define RING_CONSUMER_IDLE_OFFSET  __builtin_offsetof (RgfxRingHeader, consumer_idle)
#define RING_MASK                  (RGFX_CONTROL_RING_DATA_SIZE - 1)
#define RING_MSG_ALIGN             8

STATIC
VOID
Stall (
  IN UINTN  Microseconds
  )
{
  ((EFI_STALL)gBS->Stall)(Microseconds);
}

STATIC
VOID *
AllocateZeroPool (
  IN UINTN  Size
  )
{
  UINT8  *Memory;
  UINTN  Index;

  if (EFI_ERROR (gBS->AllocatePool (EfiBootServicesData, Size, (VOID **)&Memory))) {
    return NULL;
  }

  for (Index = 0; Index < Size; Index++) {
    Memory[Index] = 0;
  }

  return Memory;
}

STATIC
UINT32
ReadRegister (
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private,
  IN UINT64                       Offset
  )
{
  UINT32  Value;

  Value = MAX_UINT32;
  Private->PciIo->Mem.Read (Private->PciIo, EfiPciIoWidthUint32, RGFX_PCI_BAR_REGISTERS, Offset, 1, &Value);
  return Value;
}

STATIC
VOID
WriteRegister (
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private,
  IN UINT64                       Offset,
  IN UINT32                       Value
  )
{
  Private->PciIo->Mem.Write (Private->PciIo, EfiPciIoWidthUint32, RGFX_PCI_BAR_REGISTERS, Offset, 1, &Value);
}

/**
  Copy Size bytes, a multiple of 8, to or from ring memory, 8 bytes at a
  time.
**/
STATIC
VOID
CopyQwords (
  OUT volatile UINT64       *Destination,
  IN  CONST volatile UINT64 *Source,
  IN  UINTN                 Size
  )
{
  UINTN  Index;

  for (Index = 0; Index < Size / sizeof (UINT64); Index++) {
    Destination[Index] = Source[Index];
  }
}

/**
  Put a command on the submission ring, padding to the ring's end first when
  it does not fit before it, and ring the doorbell if the device's consumer
  sleeps. Message is a full message struct, its payload fields filled; its
  header is stamped here. Size is a multiple of 8.
**/
STATIC
EFI_STATUS
Submit (
  IN  REIMS_GRAPHICS_PRIVATE_DATA  *Private,
  IN  UINT32                       Type,
  IN  CONST VOID                   *Message,
  IN  UINT32                       Size,
  OUT UINT64                       *Seqno
  )
{
  volatile UINT8  *Ring;
  volatile UINT8  *Bytes;
  UINT64          Producer;
  UINT64          Consumer;
  UINT32          Offset;
  UINT32          Left;
  UINT32          Need;
  UINTN           Poll;
  UINT64          Words[8];
  RgfxMsgHeader   Pad;

  Ring     = SUBMIT_RING (Private);
  Bytes    = RING_BYTES (Ring);
  Producer = Private->SubmitProducer;
  if ((Size < sizeof (RgfxMsgHeader)) || ((Size % RING_MSG_ALIGN) != 0) || (Size > sizeof (Words))) {
    return EFI_INVALID_PARAMETER;
  }

  for (Poll = 0; ; Poll++) {
    Consumer = __atomic_load_n (RING_U64 (Ring, RING_CONSUMER_OFFSET), __ATOMIC_ACQUIRE);
    Offset   = (UINT32)Producer & RING_MASK;
    Left     = RGFX_CONTROL_RING_DATA_SIZE - Offset;
    Need     = Size + ((Left < Size) ? Left : 0);
    if (Producer - Consumer + Need <= RGFX_CONTROL_RING_DATA_SIZE) {
      break;
    }

    if (Poll == REIMS_GRAPHICS_POLL_LIMIT) {
      return EFI_TIMEOUT;
    }

    Stall (REIMS_GRAPHICS_POLL_US);
  }

  //
  // A message never wraps. The pad is a type 0 message of the bytes left;
  // with only 8 left, its type and size alone.
  //
  if (Left < Size) {
    Pad.type  = RGFX_MSG_PAD;
    Pad.size  = Left;
    Pad.seqno = 0;
    CopyQwords ((volatile UINT64 *)(Bytes + Offset), (UINT64 *)&Pad, (Left < sizeof (Pad)) ? Left : sizeof (Pad));
    Producer += Left;
    Offset    = 0;
  }

  //
  // The caller filled the payload fields; the header is stamped here.
  //
  CopyQwords (Words, (CONST UINT64 *)Message, Size);
  ((RgfxMsgHeader *)Words)->type  = Type;
  ((RgfxMsgHeader *)Words)->size  = Size;
  ((RgfxMsgHeader *)Words)->seqno = ++Private->Seqno;
  CopyQwords ((volatile UINT64 *)(Bytes + Offset), Words, Size);
  Producer += Size;

  Private->SubmitProducer = Producer;
  __atomic_store_n (RING_U64 (Ring, RING_PRODUCER_OFFSET), Producer, __ATOMIC_RELEASE);
  __atomic_thread_fence (__ATOMIC_SEQ_CST);
  if (__atomic_load_n (RING_U32 (Ring, RING_CONSUMER_IDLE_OFFSET), __ATOMIC_ACQUIRE) == 1) {
    WriteRegister (Private, RGFX_REG_DOORBELL, RGFX_CONTROL_RING_ID);
  }

  *Seqno = ((RgfxMsgHeader *)Words)->seqno;
  return EFI_SUCCESS;
}

/**
  Poll the completion ring for the answer to command Seqno: a COMPLETE, or
  a message of type Reply. Every other message is consumed and dropped. Up
  to Capacity bytes of the answer's payload go to Payload, and its type and
  payload size to Type and PayloadSize.
**/
STATIC
EFI_STATUS
WaitFor (
  IN  REIMS_GRAPHICS_PRIVATE_DATA  *Private,
  IN  UINT64                       Seqno,
  IN  UINT32                       Reply,
  OUT VOID                         *Payload,
  IN  UINT32                       Capacity,
  OUT UINT32                       *Type,
  OUT UINT32                       *PayloadSize
  )
{
  volatile UINT8        *Ring;
  volatile UINT8        *Bytes;
  UINT64                Producer;
  UINT64                Consumer;
  UINT32                Offset;
  UINT32                Left;
  UINT32          Copy;
  UINTN           Poll;
  BOOLEAN         Found;
  RgfxMsgHeader   Header;

  Ring     = EVENT_RING (Private);
  Bytes    = RING_BYTES (Ring);
  Consumer = Private->EventConsumer;
  Found    = FALSE;

  for (Poll = 0; !Found; Poll++) {
    Producer = __atomic_load_n (RING_U64 (Ring, RING_PRODUCER_OFFSET), __ATOMIC_ACQUIRE);
    while (!Found && (Consumer != Producer)) {
      Offset = (UINT32)Consumer & RING_MASK;
      Left   = RGFX_CONTROL_RING_DATA_SIZE - Offset;
      if (Left < sizeof (Header)) {
        if (Left > Producer - Consumer) {
          return EFI_DEVICE_ERROR;
        }

        Consumer += Left;
        continue;
      }

      CopyQwords ((UINT64 *)&Header, (volatile UINT64 *)(Bytes + Offset), sizeof (Header));
      if ((Header.size < ((Header.type == RGFX_MSG_PAD) ? RING_MSG_ALIGN : sizeof (Header))) ||
          ((Header.size % RING_MSG_ALIGN) != 0) ||
          (Header.size > Left) ||
          (Header.size > Producer - Consumer))
      {
        return EFI_DEVICE_ERROR;
      }

      if ((Header.type != RGFX_MSG_PAD) && (Header.seqno == Seqno) &&
          ((Header.type == RGFX_EVT_COMPLETE) || (Header.type == Reply)))
      {
        *Type        = Header.type;
        *PayloadSize = Header.size - sizeof (Header);
        Copy         = (*PayloadSize < Capacity) ? *PayloadSize : Capacity;
        CopyQwords (Payload, (volatile UINT64 *)(Bytes + Offset + sizeof (Header)), Copy & ~(UINT32)7);
        Found = TRUE;
      }

      Consumer += Header.size;
    }

    Private->EventConsumer = Consumer;
    __atomic_store_n (RING_U64 (Ring, RING_CONSUMER_OFFSET), Consumer, __ATOMIC_RELEASE);
    if (!Found) {
      if (Poll == REIMS_GRAPHICS_POLL_LIMIT) {
        return EFI_TIMEOUT;
      }

      Stall (REIMS_GRAPHICS_POLL_US);
    }
  }

  return EFI_SUCCESS;
}

/**
  Send a command answered by COMPLETE alone, and check its status.
**/
STATIC
EFI_STATUS
Command (
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private,
  IN UINT32                       Type,
  IN CONST VOID                   *Payload,
  IN UINT32                       PayloadSize
  )
{
  EFI_STATUS  Status;
  UINT64      Seqno;
  UINT64      Answer;
  UINT32      AnswerType;
  UINT32      AnswerSize;

  Status = Submit (Private, Type, Payload, PayloadSize, &Seqno);
  if (EFI_ERROR (Status)) {
    return Status;
  }

  Answer = MAX_UINT32;
  Status = WaitFor (Private, Seqno, RGFX_EVT_COMPLETE, &Answer, sizeof (Answer), &AnswerType, &AnswerSize);
  if (EFI_ERROR (Status)) {
    return Status;
  }

  if ((AnswerSize < sizeof (UINT32)) || ((UINT32)Answer != 0)) {
    return EFI_DEVICE_ERROR;
  }

  return EFI_SUCCESS;
}

EFI_STATUS
ReimsGraphicsSetScanout (
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private,
  IN BOOLEAN                      Blank
  )
{
  RgfxSetScanout  Scanout = { 0 };

  Scanout.display = Private->Display;
  Scanout.width   = Blank ? 0 : Private->Width;
  Scanout.height  = Blank ? 0 : Private->Height;
  Scanout.stride  = Blank ? 0 : Private->Stride;
  Scanout.offset  = Blank ? 0 : RGFX_ALLOCATIONS_OFFSET;
  return Command (Private, RGFX_CMD_SET_SCANOUT, &Scanout, sizeof (Scanout));
}

/**
  Ask GET_DISPLAYS for the first connected display and its preferred mode.
  The display stays 0 and the mode 0x0 when the answer names none.
**/
STATIC
VOID
ChooseMode (
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private
  )
{
  UINT64              Answer[512];
  UINT8               *Bytes;
  UINT64              Seqno;
  UINT32              Type;
  UINT32              Size;
  UINT32             Count;
  UINT32             Index;
  UINTN              Offset;
  RgfxDisplayEntry   *Display;
  RgfxModeEntry      *Mode;
  RgfxGetDisplays    GetDisplays = { 0 };

  Private->Display = 0;
  Private->Width   = 0;
  Private->Height  = 0;

  if (EFI_ERROR (Submit (Private, RGFX_CMD_GET_DISPLAYS, &GetDisplays, sizeof (GetDisplays), &Seqno)) ||
      EFI_ERROR (WaitFor (Private, Seqno, RGFX_EVT_DISPLAYS, Answer, sizeof (Answer), &Type, &Size)) ||
      (Type != RGFX_EVT_DISPLAYS))
  {
    return;
  }

  //
  // The answer's payload, after the message header: the display count, then
  // the display records at RGFX_DISPLAYS_ENTRIES_OFFSET into the message.
  //
  Bytes = (UINT8 *)Answer;
  Size  = (Size < sizeof (Answer)) ? Size : sizeof (Answer);
  if (Size < sizeof (UINT32)) {
    return;
  }

  Count  = *(UINT32 *)Bytes;
  Offset = RGFX_DISPLAYS_ENTRIES_OFFSET - sizeof (RgfxMsgHeader);
  for (Index = 0; Index < Count && Offset + sizeof (*Display) <= Size; Index++) {
    Display = (RgfxDisplayEntry *)(Bytes + Offset);
    Offset += sizeof (*Display);
    if (Index == 0) {
      Private->Display = Display->id;
    }

    if (Display->connected &&
        (Display->preferred_mode < Display->mode_count) &&
        (Offset + (UINTN)(Display->preferred_mode + 1) * sizeof (*Mode) <= Size))
    {
      Mode = (RgfxModeEntry *)(Bytes + Offset) + Display->preferred_mode;
      if ((Mode->width != 0) && (Mode->height != 0)) {
        Private->Display = Display->id;
        Private->Width   = Mode->width;
        Private->Height  = Mode->height;
        return;
      }
    }

    Offset += (UINTN)Display->mode_count * sizeof (*Mode);
  }
}

/**
  Reset the device, adopt the control rings' counters and tell the device
  the driver is ready.
**/
STATIC
VOID
OpenRings (
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private
  )
{
  WriteRegister (Private, RGFX_REG_STATUS, RGFX_STATUS_RESET);
  Private->SubmitProducer = __atomic_load_n (RING_U64 (SUBMIT_RING (Private), RING_PRODUCER_OFFSET), __ATOMIC_ACQUIRE);
  Private->EventConsumer  = __atomic_load_n (RING_U64 (EVENT_RING (Private), RING_CONSUMER_OFFSET), __ATOMIC_ACQUIRE);
  Private->Seqno          = 0;
  WriteRegister (Private, RGFX_REG_STATUS, RGFX_STATUS_DRIVER_READY);
}

/**
  The region's address: BAR 2, a memory BAR with no address translation that
  holds the size the device gives.
**/
STATIC
EFI_STATUS
FindRegion (
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private
  )
{
  EFI_ACPI_ADDRESS_SPACE_DESCRIPTOR  *Bar;
  BOOLEAN                            Usable;

  Private->RegionSize = ReadRegister (Private, RGFX_REG_REGION_SIZE_LO) |
                        ((UINT64)ReadRegister (Private, RGFX_REG_REGION_SIZE_HI) << 32);
  if (EFI_ERROR (Private->PciIo->GetBarAttributes (Private->PciIo, RGFX_PCI_BAR_REGION, NULL, (VOID **)&Bar))) {
    return EFI_UNSUPPORTED;
  }

  Private->RegionBase = Bar->AddrRangeMin;
  Usable              = (Bar->Desc == ACPI_ADDRESS_SPACE_DESCRIPTOR) &&
                        (Bar->ResType == ACPI_ADDRESS_SPACE_TYPE_MEM) &&
                        (Bar->AddrTranslationOffset == 0) &&
                        (Bar->AddrLen >= Private->RegionSize) &&
                        (Private->RegionSize > RGFX_ALLOCATIONS_OFFSET);
  gBS->FreePool (Bar);
  return Usable ? EFI_SUCCESS : EFI_UNSUPPORTED;
}

/**
  The mode's row bytes, and whether its framebuffer fits the region from
  RGFX_ALLOCATIONS_OFFSET.
**/
STATIC
BOOLEAN
ModeFits (
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private
  )
{
  UINT64  Stride;

  Stride = ((UINT64)Private->Width * 4 + REIMS_GRAPHICS_STRIDE_ALIGN - 1) &
           ~(UINT64)(REIMS_GRAPHICS_STRIDE_ALIGN - 1);
  if ((Private->Width == 0) || (Private->Height == 0) || (Stride > MAX_UINT32) ||
      (Stride * Private->Height > Private->RegionSize - RGFX_ALLOCATIONS_OFFSET))
  {
    return FALSE;
  }

  Private->Stride = (UINT32)Stride;
  return TRUE;
}

STATIC
REIMS_GRAPHICS_PRIVATE_DATA *
FindController (
  IN EFI_HANDLE  Controller
  )
{
  REIMS_GRAPHICS_PRIVATE_DATA  *Private;

  for (Private = mControllers; Private != NULL && Private->Controller != Controller; Private = Private->Next) {
  }

  return Private;
}

/**
  Check whether Controller is the reims-graphics device.
**/
EFI_STATUS
EFIAPI
ReimsGraphicsDriverBindingSupported (
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN EFI_HANDLE                   Controller,
  IN EFI_DEVICE_PATH_PROTOCOL     *RemainingDevicePath
  )
{
  EFI_STATUS           Status;
  EFI_PCI_IO_PROTOCOL  *PciIo;
  PCI_TYPE00           Pci;

  Status = gBS->OpenProtocol (
                  Controller,
                  &gEfiPciIoProtocolGuid,
                  (VOID **)&PciIo,
                  This->DriverBindingHandle,
                  Controller,
                  EFI_OPEN_PROTOCOL_BY_DRIVER
                  );
  if (EFI_ERROR (Status)) {
    return Status;
  }

  Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0, sizeof (Pci) / sizeof (UINT32), &Pci);
  if (!EFI_ERROR (Status) &&
      ((Pci.Hdr.VendorId != RGFX_PCI_VENDOR_ID) || (Pci.Hdr.DeviceId != RGFX_PCI_DEVICE_ID)))
  {
    Status = EFI_UNSUPPORTED;
  }

  gBS->CloseProtocol (Controller, &gEfiPciIoProtocolGuid, This->DriverBindingHandle, Controller);
  return Status;
}

/**
  Undo Start's steps on the controller: the PCI attributes it changed and
  its PCI I/O open.
**/
STATIC
VOID
ReleaseController (
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private,
  IN BOOLEAN                      RestoreAttributes
  )
{
  if (RestoreAttributes) {
    Private->PciIo->Attributes (
                      Private->PciIo,
                      EfiPciIoAttributeOperationSet,
                      Private->OriginalPciAttributes,
                      NULL
                      );
  }

  gBS->CloseProtocol (Private->Controller, &gEfiPciIoProtocolGuid, This->DriverBindingHandle, Private->Controller);
  gBS->FreePool (Private);
}

/**
  Append Text to Line at Length, as far as Line's Size characters leave room
  for the terminating null; return the new length.
**/
STATIC
UINTN
AppendText (
  IN OUT CHAR16  *Line,
  IN     UINTN   Size,
  IN     UINTN   Length,
  IN     CHAR8   *Text
  )
{
  while ((*Text != 0) && (Length + 1 < Size)) {
    Line[Length++] = (CHAR16)*Text++;
  }

  Line[Length] = 0;
  return Length;
}

/**
  Append Value in decimal to Line at Length, as AppendText does.
**/
STATIC
UINTN
AppendDecimal (
  IN OUT CHAR16  *Line,
  IN     UINTN   Size,
  IN     UINTN   Length,
  IN     UINT32  Value
  )
{
  CHAR8  Digits[11];
  UINTN  Index;

  Index         = sizeof (Digits) - 1;
  Digits[Index] = 0;
  do {
    Digits[--Index] = (CHAR8)('0' + Value % 10);
    Value          /= 10;
  } while (Value != 0);

  return AppendText (Line, Size, Length, &Digits[Index]);
}

/**
  Say on the firmware's console that the device speaks version Version of the
  wire, not this driver's RGFX_VERSION, so the driver leaves it: the version
  names the region's layout and the messages, and a driver of another version
  misreads them (docs/graphics-kext/design.md, "The wire's version"). Firmware
  with no console yet says nothing.
**/
STATIC
VOID
SayWrongVersion (
  IN UINT32  Version
  )
{
  CHAR16  Line[160];
  UINTN   Length;

  if ((mSystemTable == NULL) || (mSystemTable->ConOut == NULL)) {
    return;
  }

  Length = AppendText (Line, ARRAY_SIZE (Line), 0, "ReimsGraphics: the device speaks wire version ");
  Length = AppendDecimal (Line, ARRAY_SIZE (Line), Length, Version);
  Length = AppendText (Line, ARRAY_SIZE (Line), Length, ", this option ROM version ");
  Length = AppendDecimal (Line, ARRAY_SIZE (Line), Length, RGFX_VERSION);
  Length = AppendText (Line, ARRAY_SIZE (Line), Length, ": no display. The ROM and the host must come from one build.\r\n");
  mSystemTable->ConOut->OutputString (mSystemTable->ConOut, Line);
}

/**
  Bind to Controller: check the device, open its rings, choose the mode, and
  install Device Path and Graphics Output protocols on a child handle with
  that mode set.
**/
EFI_STATUS
EFIAPI
ReimsGraphicsDriverBindingStart (
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN EFI_HANDLE                   Controller,
  IN EFI_DEVICE_PATH_PROTOCOL     *RemainingDevicePath
  )
{
  EFI_STATUS                   Status;
  REIMS_GRAPHICS_PRIVATE_DATA  *Private;
  EFI_DEVICE_PATH_PROTOCOL     *ParentDevicePath;
  ACPI_ADR_DEVICE_PATH         AcpiDeviceNode;
  EFI_PCI_IO_PROTOCOL          *ChildPciIo;

  Private = AllocateZeroPool (sizeof (*Private));
  if (Private == NULL) {
    return EFI_OUT_OF_RESOURCES;
  }

  Private->Signature  = REIMS_GRAPHICS_PRIVATE_DATA_SIGNATURE;
  Private->Controller = Controller;

  Status = gBS->OpenProtocol (
                  Controller,
                  &gEfiPciIoProtocolGuid,
                  (VOID **)&Private->PciIo,
                  This->DriverBindingHandle,
                  Controller,
                  EFI_OPEN_PROTOCOL_BY_DRIVER
                  );
  if (EFI_ERROR (Status)) {
    gBS->FreePool (Private);
    return Status;
  }

  Status = Private->PciIo->Attributes (
                             Private->PciIo,
                             EfiPciIoAttributeOperationGet,
                             0,
                             &Private->OriginalPciAttributes
                             );
  if (EFI_ERROR (Status)) {
    ReleaseController (This, Private, FALSE);
    return Status;
  }

  Status = Private->PciIo->Attributes (
                             Private->PciIo,
                             EfiPciIoAttributeOperationEnable,
                             EFI_PCI_IO_ATTRIBUTE_MEMORY | EFI_PCI_IO_ATTRIBUTE_BUS_MASTER,
                             NULL
                             );
  if (!EFI_ERROR (Status) && (ReadRegister (Private, RGFX_REG_MAGIC) != RGFX_MAGIC)) {
    Status = EFI_UNSUPPORTED;
  }

  if (!EFI_ERROR (Status) && (ReadRegister (Private, RGFX_REG_VERSION) != RGFX_VERSION)) {
    SayWrongVersion (ReadRegister (Private, RGFX_REG_VERSION));
    Status = EFI_UNSUPPORTED;
  }

  if (!EFI_ERROR (Status)) {
    Status = FindRegion (Private);
  }

  if (!EFI_ERROR (Status)) {
    OpenRings (Private);
    ChooseMode (Private);
    if (!ModeFits (Private)) {
      Private->Width  = REIMS_GRAPHICS_FALLBACK_WIDTH;
      Private->Height = REIMS_GRAPHICS_FALLBACK_HEIGHT;
      if (!ModeFits (Private)) {
        Status = EFI_UNSUPPORTED;
      }
    }
  }

  if (!EFI_ERROR (Status)) {
    Status = gBS->HandleProtocol (Controller, &gEfiDevicePathProtocolGuid, (VOID **)&ParentDevicePath);
  }

  if (EFI_ERROR (Status)) {
    ReleaseController (This, Private, TRUE);
    return Status;
  }

  //
  // The Graphics Output child is an ACPI _ADR node below the controller.
  //
  AcpiDeviceNode.Header.Type      = ACPI_DEVICE_PATH;
  AcpiDeviceNode.Header.SubType   = ACPI_ADR_DP;
  AcpiDeviceNode.ADR              = ACPI_DISPLAY_ADR (1, 0, 0, 1, 0, ACPI_ADR_DISPLAY_TYPE_OTHER, 0, 0);
  AcpiDeviceNode.Header.Length[0] = (UINT8)sizeof (ACPI_ADR_DEVICE_PATH);
  AcpiDeviceNode.Header.Length[1] = 0;
  Private->GopDevicePath          = mDevicePathUtilities->AppendDeviceNode (ParentDevicePath, &AcpiDeviceNode.Header);
  if (Private->GopDevicePath == NULL) {
    ReleaseController (This, Private, TRUE);
    return EFI_OUT_OF_RESOURCES;
  }

  ReimsGraphicsGraphicsOutputInit (Private);
  Status = Private->GraphicsOutput.SetMode (&Private->GraphicsOutput, 0);
  if (!EFI_ERROR (Status)) {
    Status = gBS->InstallMultipleProtocolInterfaces (
                    &Private->Handle,
                    &gEfiDevicePathProtocolGuid,
                    Private->GopDevicePath,
                    &gEfiGraphicsOutputProtocolGuid,
                    &Private->GraphicsOutput,
                    NULL
                    );
  }

  if (EFI_ERROR (Status)) {
    gBS->FreePool (Private->GopDevicePath);
    ReleaseController (This, Private, TRUE);
    return Status;
  }

  gBS->OpenProtocol (
         Controller,
         &gEfiPciIoProtocolGuid,
         (VOID **)&ChildPciIo,
         This->DriverBindingHandle,
         Private->Handle,
         EFI_OPEN_PROTOCOL_BY_CHILD_CONTROLLER
         );

  Private->Next = mControllers;
  mControllers  = Private;
  return EFI_SUCCESS;
}

/**
  Remove the Graphics Output child, then release the controller. The display
  keeps its scanout, so the frame stays up for whatever drives it next.
**/
EFI_STATUS
EFIAPI
ReimsGraphicsDriverBindingStop (
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN EFI_HANDLE                   Controller,
  IN UINTN                        NumberOfChildren,
  IN EFI_HANDLE                   *ChildHandleBuffer
  )
{
  REIMS_GRAPHICS_PRIVATE_DATA  **Link;
  REIMS_GRAPHICS_PRIVATE_DATA  *Private;
  EFI_STATUS                   Status;

  Private = FindController (Controller);
  if (Private == NULL) {
    return EFI_SUCCESS;
  }

  if (Private->Handle != NULL) {
    gBS->CloseProtocol (Controller, &gEfiPciIoProtocolGuid, This->DriverBindingHandle, Private->Handle);
    Status = gBS->UninstallMultipleProtocolInterfaces (
                    Private->Handle,
                    &gEfiDevicePathProtocolGuid,
                    Private->GopDevicePath,
                    &gEfiGraphicsOutputProtocolGuid,
                    &Private->GraphicsOutput,
                    NULL
                    );
    if (EFI_ERROR (Status)) {
      return Status;
    }

    gBS->FreePool (Private->GopDevicePath);
    Private->Handle = NULL;
  }

  if (NumberOfChildren != 0) {
    return EFI_SUCCESS;
  }

  for (Link = &mControllers; *Link != Private; Link = &(*Link)->Next) {
  }

  *Link = Private->Next;
  ReleaseController (This, Private, TRUE);
  return EFI_SUCCESS;
}

EFI_DRIVER_BINDING_PROTOCOL  gReimsGraphicsDriverBinding = {
  ReimsGraphicsDriverBindingSupported,
  ReimsGraphicsDriverBindingStart,
  ReimsGraphicsDriverBindingStop,
  0x10,
  NULL,
  NULL
};

/**
  Image entry point: install the driver binding and component names on the
  image handle.
**/
EFI_STATUS
EFIAPI
ReimsGraphicsEntryPoint (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS  Status;

  gImageHandle = ImageHandle;
  gBS          = SystemTable->BootServices;
  mSystemTable = SystemTable;

  Status = gBS->LocateProtocol (&gEfiDevicePathUtilitiesProtocolGuid, NULL, (VOID **)&mDevicePathUtilities);
  if (EFI_ERROR (Status)) {
    return Status;
  }

  gReimsGraphicsDriverBinding.ImageHandle         = ImageHandle;
  gReimsGraphicsDriverBinding.DriverBindingHandle = ImageHandle;
  return gBS->InstallMultipleProtocolInterfaces (
                &gReimsGraphicsDriverBinding.DriverBindingHandle,
                &gEfiDriverBindingProtocolGuid,
                &gReimsGraphicsDriverBinding,
                &gEfiComponentNameProtocolGuid,
                &gReimsGraphicsComponentName,
                &gEfiComponentName2ProtocolGuid,
                &gReimsGraphicsComponentName2,
                NULL
                );
}
