/** @file
  The subset of the UEFI specification this driver uses: base types, the
  system and boot services tables, the PCI configuration header, device path
  nodes, and the driver model, PCI I/O, Graphics Output and Simple Text
  Output protocols.

  Names follow the UEFI specification. Table services the driver does not
  call are declared as opaque pointers so that every field keeps its
  specified offset. The build targets x86_64-unknown-windows, whose default
  calling convention is already the Microsoft x64 convention UEFI requires,
  so EFIAPI needs no attribute.
**/

#ifndef UEFI_H_
#define UEFI_H_

//
// Base types and macros.
//
typedef unsigned long long  UINT64;
typedef long long           INT64;
typedef unsigned int        UINT32;
typedef int                 INT32;
typedef unsigned short      UINT16;
typedef unsigned short      CHAR16;
typedef short               INT16;
typedef unsigned char       BOOLEAN;
typedef unsigned char       UINT8;
typedef char                CHAR8;
typedef signed char         INT8;
typedef UINT64              UINTN;
typedef INT64               INTN;

#define VOID      void
#define CONST     const
#define STATIC    static
#define IN
#define OUT
#define OPTIONAL
#define EFIAPI

#define TRUE   ((BOOLEAN)(1 == 1))
#define FALSE  ((BOOLEAN)(0 == 1))
#define NULL   ((VOID *) 0)

#define MAX_BIT     0x8000000000000000ULL
#define MAX_UINT32  ((UINT32)0xFFFFFFFF)

#define OFFSET_OF(TYPE, Field)  ((UINTN) __builtin_offsetof(TYPE, Field))
#define BASE_CR(Record, TYPE, Field) \
  ((TYPE *) ((CHAR8 *) (Record) - OFFSET_OF (TYPE, Field)))

#define ARRAY_SIZE(Array)  (sizeof (Array) / sizeof ((Array)[0]))
#define MAX(a, b)          (((a) > (b)) ? (a) : (b))

#define SIGNATURE_16(A, B)        ((A) | (B << 8))
#define SIGNATURE_32(A, B, C, D)  (SIGNATURE_16 (A, B) | (SIGNATURE_16 (C, D) << 16))

//
// Status codes and common types.
//
typedef UINTN   EFI_STATUS;
typedef VOID    *EFI_HANDLE;
typedef UINT64  EFI_PHYSICAL_ADDRESS;

typedef struct {
  UINT32    Data1;
  UINT16    Data2;
  UINT16    Data3;
  UINT8     Data4[8];
} EFI_GUID;

#define EFIERR(StatusCode)  ((EFI_STATUS)(MAX_BIT | (StatusCode)))
#define EFI_ERROR(Status)   (((INTN)(EFI_STATUS)(Status)) < 0)

#define EFI_SUCCESS            0
#define EFI_INVALID_PARAMETER  EFIERR (2)
#define EFI_UNSUPPORTED        EFIERR (3)
#define EFI_OUT_OF_RESOURCES   EFIERR (9)
#define EFI_ALREADY_STARTED    EFIERR (20)

#define SIZE_4GB  0x100000000ULL

#define EFI_PAGE_SIZE   0x1000
#define EFI_PAGE_MASK   0xFFF
#define EFI_PAGE_SHIFT  12

#define EFI_SIZE_TO_PAGES(Size) \
  (((Size) >> EFI_PAGE_SHIFT) + (((Size) & EFI_PAGE_MASK) ? 1 : 0))

#define EFI_2_10_SYSTEM_TABLE_REVISION  ((2 << 16) | (10))

typedef enum {
  AllocateAnyPages,
  AllocateMaxAddress,
  AllocateAddress,
  MaxAllocateType
} EFI_ALLOCATE_TYPE;

typedef enum {
  EfiReservedMemoryType,
  EfiLoaderCode,
  EfiLoaderData,
  EfiBootServicesCode,
  EfiBootServicesData,
  EfiRuntimeServicesCode,
  EfiRuntimeServicesData,
  EfiConventionalMemory,
  EfiUnusableMemory,
  EfiACPIReclaimMemory,
  EfiACPIMemoryNVS,
  EfiMemoryMappedIO,
  EfiMemoryMappedIOPortSpace,
  EfiPalCode,
  EfiPersistentMemory,
  EfiUnacceptedMemoryType,
  EfiMaxMemoryType
} EFI_MEMORY_TYPE;

//
// System table and boot services.
//
typedef struct {
  UINT64    Signature;
  UINT32    Revision;
  UINT32    HeaderSize;
  UINT32    CRC32;
  UINT32    Reserved;
} EFI_TABLE_HEADER;

#define EFI_OPEN_PROTOCOL_BY_HANDLE_PROTOCOL   0x00000001
#define EFI_OPEN_PROTOCOL_GET_PROTOCOL         0x00000002
#define EFI_OPEN_PROTOCOL_TEST_PROTOCOL        0x00000004
#define EFI_OPEN_PROTOCOL_BY_CHILD_CONTROLLER  0x00000008
#define EFI_OPEN_PROTOCOL_BY_DRIVER            0x00000010
#define EFI_OPEN_PROTOCOL_EXCLUSIVE            0x00000020

typedef
EFI_STATUS
(EFIAPI *EFI_ALLOCATE_POOL)(
  IN  EFI_MEMORY_TYPE  PoolType,
  IN  UINTN            Size,
  OUT VOID             **Buffer
  );

typedef
EFI_STATUS
(EFIAPI *EFI_FREE_POOL)(
  IN  VOID  *Buffer
  );

typedef
EFI_STATUS
(EFIAPI *EFI_HANDLE_PROTOCOL)(
  IN  EFI_HANDLE  Handle,
  IN  EFI_GUID    *Protocol,
  OUT VOID        **Interface
  );

typedef
EFI_STATUS
(EFIAPI *EFI_OPEN_PROTOCOL)(
  IN  EFI_HANDLE  Handle,
  IN  EFI_GUID    *Protocol,
  OUT VOID        **Interface  OPTIONAL,
  IN  EFI_HANDLE  AgentHandle,
  IN  EFI_HANDLE  ControllerHandle,
  IN  UINT32      Attributes
  );

typedef
EFI_STATUS
(EFIAPI *EFI_CLOSE_PROTOCOL)(
  IN EFI_HANDLE  Handle,
  IN EFI_GUID    *Protocol,
  IN EFI_HANDLE  AgentHandle,
  IN EFI_HANDLE  ControllerHandle
  );

typedef
EFI_STATUS
(EFIAPI *EFI_LOCATE_PROTOCOL)(
  IN  EFI_GUID  *Protocol,
  IN  VOID      *Registration  OPTIONAL,
  OUT VOID      **Interface
  );

typedef
EFI_STATUS
(EFIAPI *EFI_INSTALL_MULTIPLE_PROTOCOL_INTERFACES)(
  IN OUT EFI_HANDLE  *Handle,
  ...
  );

typedef
EFI_STATUS
(EFIAPI *EFI_UNINSTALL_MULTIPLE_PROTOCOL_INTERFACES)(
  IN EFI_HANDLE  Handle,
  ...
  );

typedef struct {
  EFI_TABLE_HEADER                              Hdr;
  VOID                                          *RaiseTPL;
  VOID                                          *RestoreTPL;
  VOID                                          *AllocatePages;
  VOID                                          *FreePages;
  VOID                                          *GetMemoryMap;
  EFI_ALLOCATE_POOL                             AllocatePool;
  EFI_FREE_POOL                                 FreePool;
  VOID                                          *CreateEvent;
  VOID                                          *SetTimer;
  VOID                                          *WaitForEvent;
  VOID                                          *SignalEvent;
  VOID                                          *CloseEvent;
  VOID                                          *CheckEvent;
  VOID                                          *InstallProtocolInterface;
  VOID                                          *ReinstallProtocolInterface;
  VOID                                          *UninstallProtocolInterface;
  EFI_HANDLE_PROTOCOL                           HandleProtocol;
  VOID                                          *Reserved;
  VOID                                          *RegisterProtocolNotify;
  VOID                                          *LocateHandle;
  VOID                                          *LocateDevicePath;
  VOID                                          *InstallConfigurationTable;
  VOID                                          *LoadImage;
  VOID                                          *StartImage;
  VOID                                          *Exit;
  VOID                                          *UnloadImage;
  VOID                                          *ExitBootServices;
  VOID                                          *GetNextMonotonicCount;
  VOID                                          *Stall;
  VOID                                          *SetWatchdogTimer;
  VOID                                          *ConnectController;
  VOID                                          *DisconnectController;
  EFI_OPEN_PROTOCOL                             OpenProtocol;
  EFI_CLOSE_PROTOCOL                            CloseProtocol;
  VOID                                          *OpenProtocolInformation;
  VOID                                          *ProtocolsPerHandle;
  VOID                                          *LocateHandleBuffer;
  EFI_LOCATE_PROTOCOL                           LocateProtocol;
  EFI_INSTALL_MULTIPLE_PROTOCOL_INTERFACES      InstallMultipleProtocolInterfaces;
  EFI_UNINSTALL_MULTIPLE_PROTOCOL_INTERFACES    UninstallMultipleProtocolInterfaces;
  VOID                                          *CalculateCrc32;
  VOID                                          *CopyMem;
  VOID                                          *SetMem;
  VOID                                          *CreateEventEx;
} EFI_BOOT_SERVICES;

//
// The Simple Text Output protocol, up to the one service the driver calls.
//
typedef struct _EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL;

typedef
EFI_STATUS
(EFIAPI *EFI_TEXT_STRING)(
  IN EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL  *This,
  IN CHAR16                           *String
  );

struct _EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL {
  VOID               *Reset;
  EFI_TEXT_STRING    OutputString;
};

typedef struct {
  EFI_TABLE_HEADER                   Hdr;
  CHAR16                             *FirmwareVendor;
  UINT32                             FirmwareRevision;
  EFI_HANDLE                         ConsoleInHandle;
  VOID                               *ConIn;
  EFI_HANDLE                         ConsoleOutHandle;
  EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL    *ConOut;
  EFI_HANDLE           StandardErrorHandle;
  VOID                 *StdErr;
  VOID                 *RuntimeServices;
  EFI_BOOT_SERVICES    *BootServices;
  UINTN                NumberOfTableEntries;
  VOID                 *ConfigurationTable;
} EFI_SYSTEM_TABLE;

//
// PCI configuration space for a type 0 (device) header.
//
typedef struct {
  UINT16    VendorId;
  UINT16    DeviceId;
  UINT16    Command;
  UINT16    Status;
  UINT8     RevisionID;
  UINT8     ClassCode[3];
  UINT8     CacheLineSize;
  UINT8     LatencyTimer;
  UINT8     HeaderType;
  UINT8     BIST;
} PCI_DEVICE_INDEPENDENT_REGION;

typedef struct {
  UINT32    Bar[6];
  UINT32    CISPtr;
  UINT16    SubsystemVendorID;
  UINT16    SubsystemID;
  UINT32    ExpansionRomBar;
  UINT8     CapabilityPtr;
  UINT8     Reserved1[3];
  UINT32    Reserved2;
  UINT8     InterruptLine;
  UINT8     InterruptPin;
  UINT8     MinGnt;
  UINT8     MaxLat;
} PCI_DEVICE_HEADER_TYPE_REGION;

typedef struct {
  PCI_DEVICE_INDEPENDENT_REGION    Hdr;
  PCI_DEVICE_HEADER_TYPE_REGION    Device;
} PCI_TYPE00;

//
// Device Path protocol and the ACPI _ADR node (UEFI specification, 10.2
// and 10.3.3), and the Device Path Utilities protocol (10.5).
//
#define EFI_DEVICE_PATH_PROTOCOL_GUID \
  { 0x09576e91, 0x6d3f, 0x11d2, { 0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b } }

typedef struct {
  UINT8    Type;
  UINT8    SubType;
  UINT8    Length[2];
} EFI_DEVICE_PATH_PROTOCOL;

#define ACPI_DEVICE_PATH  0x02
#define ACPI_ADR_DP       0x03

typedef struct {
  EFI_DEVICE_PATH_PROTOCOL    Header;
  UINT32                      ADR;
} ACPI_ADR_DEVICE_PATH;

#define ACPI_ADR_DISPLAY_TYPE_OTHER  0

#define ACPI_DISPLAY_ADR(_DeviceIdScheme, _HeadId, _NonVgaOutput, _BiosCanDetect, _VendorInfo, _Type, _Port, _Index) \
  ((UINT32)(((UINT32)((_DeviceIdScheme) & 0x1) << 31) |                     \
            (((_HeadId) & 0x7) << 18) |                                    \
            (((_NonVgaOutput) & 0x1) << 17) |                              \
            (((_BiosCanDetect) & 0x1) << 16) |                             \
            (((_VendorInfo) & 0xf) << 12) |                                \
            (((_Type) & 0xf) << 8) |                                       \
            (((_Port) & 0xf) << 4) |                                       \
            ((_Index) & 0xf)))

#define EFI_DEVICE_PATH_UTILITIES_PROTOCOL_GUID \
  { 0x0379be4e, 0xd706, 0x437d, { 0xb0, 0x37, 0xed, 0xb8, 0x2f, 0xb7, 0x72, 0xa4 } }

typedef
EFI_DEVICE_PATH_PROTOCOL *
(EFIAPI *EFI_DEVICE_PATH_UTILS_APPEND_NODE)(
  IN CONST EFI_DEVICE_PATH_PROTOCOL  *DevicePath,
  IN CONST EFI_DEVICE_PATH_PROTOCOL  *DeviceNode
  );

typedef struct {
  VOID                                 *GetDevicePathSize;
  VOID                                 *DuplicateDevicePath;
  VOID                                 *AppendDevicePath;
  EFI_DEVICE_PATH_UTILS_APPEND_NODE    AppendDeviceNode;
  VOID                                 *AppendDevicePathInstance;
  VOID                                 *GetNextDevicePathInstance;
  VOID                                 *IsDevicePathMultiInstance;
  VOID                                 *CreateDeviceNode;
} EFI_DEVICE_PATH_UTILITIES_PROTOCOL;

//
// Driver Binding protocol (UEFI specification, 11.1).
//
#define EFI_DRIVER_BINDING_PROTOCOL_GUID \
  { 0x18a031ab, 0xb443, 0x4d1a, { 0xa5, 0xc0, 0x0c, 0x09, 0x26, 0x1e, 0x9f, 0x71 } }

typedef struct _EFI_DRIVER_BINDING_PROTOCOL EFI_DRIVER_BINDING_PROTOCOL;

typedef
EFI_STATUS
(EFIAPI *EFI_DRIVER_BINDING_SUPPORTED)(
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN EFI_HANDLE                   ControllerHandle,
  IN EFI_DEVICE_PATH_PROTOCOL     *RemainingDevicePath OPTIONAL
  );

typedef
EFI_STATUS
(EFIAPI *EFI_DRIVER_BINDING_START)(
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN EFI_HANDLE                   ControllerHandle,
  IN EFI_DEVICE_PATH_PROTOCOL     *RemainingDevicePath OPTIONAL
  );

typedef
EFI_STATUS
(EFIAPI *EFI_DRIVER_BINDING_STOP)(
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN EFI_HANDLE                   ControllerHandle,
  IN UINTN                        NumberOfChildren,
  IN EFI_HANDLE                   *ChildHandleBuffer OPTIONAL
  );

struct _EFI_DRIVER_BINDING_PROTOCOL {
  EFI_DRIVER_BINDING_SUPPORTED    Supported;
  EFI_DRIVER_BINDING_START        Start;
  EFI_DRIVER_BINDING_STOP         Stop;
  UINT32                          Version;
  EFI_HANDLE                      ImageHandle;
  EFI_HANDLE                      DriverBindingHandle;
};

//
// Component Name (ISO 639-2 language codes) and Component Name 2 (RFC 4646
// language codes) protocols (UEFI specification, 11.5).
//
#define EFI_COMPONENT_NAME_PROTOCOL_GUID \
  { 0x107a772c, 0xd5e1, 0x11d4, { 0x9a, 0x46, 0x00, 0x90, 0x27, 0x3f, 0xc1, 0x4d } }

typedef struct _EFI_COMPONENT_NAME_PROTOCOL EFI_COMPONENT_NAME_PROTOCOL;

typedef
EFI_STATUS
(EFIAPI *EFI_COMPONENT_NAME_GET_DRIVER_NAME)(
  IN  EFI_COMPONENT_NAME_PROTOCOL  *This,
  IN  CHAR8                        *Language,
  OUT CHAR16                       **DriverName
  );

typedef
EFI_STATUS
(EFIAPI *EFI_COMPONENT_NAME_GET_CONTROLLER_NAME)(
  IN  EFI_COMPONENT_NAME_PROTOCOL  *This,
  IN  EFI_HANDLE                   ControllerHandle,
  IN  EFI_HANDLE                   ChildHandle OPTIONAL,
  IN  CHAR8                        *Language,
  OUT CHAR16                       **ControllerName
  );

struct _EFI_COMPONENT_NAME_PROTOCOL {
  EFI_COMPONENT_NAME_GET_DRIVER_NAME        GetDriverName;
  EFI_COMPONENT_NAME_GET_CONTROLLER_NAME    GetControllerName;
  CHAR8                                     *SupportedLanguages;
};

#define EFI_COMPONENT_NAME2_PROTOCOL_GUID \
  { 0x6a7a5cff, 0xe8d9, 0x4f70, { 0xba, 0xda, 0x75, 0xab, 0x30, 0x25, 0xce, 0x14 } }

typedef struct _EFI_COMPONENT_NAME2_PROTOCOL EFI_COMPONENT_NAME2_PROTOCOL;

typedef
EFI_STATUS
(EFIAPI *EFI_COMPONENT_NAME2_GET_DRIVER_NAME)(
  IN  EFI_COMPONENT_NAME2_PROTOCOL  *This,
  IN  CHAR8                         *Language,
  OUT CHAR16                        **DriverName
  );

typedef
EFI_STATUS
(EFIAPI *EFI_COMPONENT_NAME2_GET_CONTROLLER_NAME)(
  IN  EFI_COMPONENT_NAME2_PROTOCOL  *This,
  IN  EFI_HANDLE                    ControllerHandle,
  IN  EFI_HANDLE                    ChildHandle OPTIONAL,
  IN  CHAR8                         *Language,
  OUT CHAR16                        **ControllerName
  );

struct _EFI_COMPONENT_NAME2_PROTOCOL {
  EFI_COMPONENT_NAME2_GET_DRIVER_NAME        GetDriverName;
  EFI_COMPONENT_NAME2_GET_CONTROLLER_NAME    GetControllerName;
  CHAR8                                      *SupportedLanguages;
};

//
// Driver Supported EFI Version protocol (UEFI specification, 11.6).
//
#define EFI_DRIVER_SUPPORTED_EFI_VERSION_PROTOCOL_GUID \
  { 0x5c198761, 0x16a8, 0x4e69, { 0x97, 0x2c, 0x89, 0xd6, 0x79, 0x54, 0xf8, 0x1d } }

typedef struct {
  UINT32    Length;
  UINT32    FirmwareVersion;
} EFI_DRIVER_SUPPORTED_EFI_VERSION_PROTOCOL;

//
// PCI I/O protocol (UEFI specification, 14.4).
//
#define EFI_PCI_IO_PROTOCOL_GUID \
  { 0x4cf5b200, 0x68b8, 0x4ca5, { 0x9e, 0xec, 0xb2, 0x3e, 0x3f, 0x50, 0x02, 0x9a } }

typedef struct _EFI_PCI_IO_PROTOCOL EFI_PCI_IO_PROTOCOL;

typedef enum {
  EfiPciIoWidthUint8,
  EfiPciIoWidthUint16,
  EfiPciIoWidthUint32,
  EfiPciIoWidthUint64,
  EfiPciIoWidthFifoUint8,
  EfiPciIoWidthFifoUint16,
  EfiPciIoWidthFifoUint32,
  EfiPciIoWidthFifoUint64,
  EfiPciIoWidthFillUint8,
  EfiPciIoWidthFillUint16,
  EfiPciIoWidthFillUint32,
  EfiPciIoWidthFillUint64,
  EfiPciIoWidthMaximum
} EFI_PCI_IO_PROTOCOL_WIDTH;

typedef enum {
  EfiPciIoOperationBusMasterRead,
  EfiPciIoOperationBusMasterWrite,
  EfiPciIoOperationBusMasterCommonBuffer,
  EfiPciIoOperationMaximum
} EFI_PCI_IO_PROTOCOL_OPERATION;

typedef enum {
  EfiPciIoAttributeOperationGet,
  EfiPciIoAttributeOperationSet,
  EfiPciIoAttributeOperationEnable,
  EfiPciIoAttributeOperationDisable,
  EfiPciIoAttributeOperationSupported,
  EfiPciIoAttributeOperationMaximum
} EFI_PCI_IO_PROTOCOL_ATTRIBUTE_OPERATION;

#define EFI_PCI_IO_ATTRIBUTE_IO          0x0100
#define EFI_PCI_IO_ATTRIBUTE_MEMORY      0x0200
#define EFI_PCI_IO_ATTRIBUTE_BUS_MASTER  0x0400

#define EFI_PCI_DEVICE_ENABLE \
  (EFI_PCI_IO_ATTRIBUTE_IO | EFI_PCI_IO_ATTRIBUTE_MEMORY | EFI_PCI_IO_ATTRIBUTE_BUS_MASTER)

typedef
EFI_STATUS
(EFIAPI *EFI_PCI_IO_PROTOCOL_MEM)(
  IN     EFI_PCI_IO_PROTOCOL        *This,
  IN     EFI_PCI_IO_PROTOCOL_WIDTH  Width,
  IN     UINT8                      BarIndex,
  IN     UINT64                     Offset,
  IN     UINTN                      Count,
  IN OUT VOID                       *Buffer
  );

typedef
EFI_STATUS
(EFIAPI *EFI_PCI_IO_PROTOCOL_CONFIG)(
  IN     EFI_PCI_IO_PROTOCOL        *This,
  IN     EFI_PCI_IO_PROTOCOL_WIDTH  Width,
  IN     UINT32                     Offset,
  IN     UINTN                      Count,
  IN OUT VOID                       *Buffer
  );

typedef struct {
  EFI_PCI_IO_PROTOCOL_MEM    Read;
  EFI_PCI_IO_PROTOCOL_MEM    Write;
} EFI_PCI_IO_PROTOCOL_ACCESS;

typedef struct {
  EFI_PCI_IO_PROTOCOL_CONFIG    Read;
  EFI_PCI_IO_PROTOCOL_CONFIG    Write;
} EFI_PCI_IO_PROTOCOL_CONFIG_ACCESS;

typedef
EFI_STATUS
(EFIAPI *EFI_PCI_IO_PROTOCOL_MAP)(
  IN     EFI_PCI_IO_PROTOCOL            *This,
  IN     EFI_PCI_IO_PROTOCOL_OPERATION  Operation,
  IN     VOID                           *HostAddress,
  IN OUT UINTN                          *NumberOfBytes,
  OUT    EFI_PHYSICAL_ADDRESS           *DeviceAddress,
  OUT    VOID                           **Mapping
  );

typedef
EFI_STATUS
(EFIAPI *EFI_PCI_IO_PROTOCOL_UNMAP)(
  IN EFI_PCI_IO_PROTOCOL  *This,
  IN VOID                 *Mapping
  );

typedef
EFI_STATUS
(EFIAPI *EFI_PCI_IO_PROTOCOL_ALLOCATE_BUFFER)(
  IN     EFI_PCI_IO_PROTOCOL  *This,
  IN     EFI_ALLOCATE_TYPE    Type,
  IN     EFI_MEMORY_TYPE      MemoryType,
  IN     UINTN                Pages,
  OUT    VOID                 **HostAddress,
  IN     UINT64               Attributes
  );

typedef
EFI_STATUS
(EFIAPI *EFI_PCI_IO_PROTOCOL_FREE_BUFFER)(
  IN EFI_PCI_IO_PROTOCOL  *This,
  IN UINTN                Pages,
  IN VOID                 *HostAddress
  );

typedef
EFI_STATUS
(EFIAPI *EFI_PCI_IO_PROTOCOL_ATTRIBUTES)(
  IN     EFI_PCI_IO_PROTOCOL                      *This,
  IN     EFI_PCI_IO_PROTOCOL_ATTRIBUTE_OPERATION  Operation,
  IN     UINT64                                   Attributes,
  OUT    UINT64                                   *Result OPTIONAL
  );

//
// Resources is pool memory holding an ACPI address space descriptor for the
// BAR, then an end tag; the caller frees it.
//
typedef
EFI_STATUS
(EFIAPI *EFI_PCI_IO_PROTOCOL_GET_BAR_ATTRIBUTES)(
  IN  EFI_PCI_IO_PROTOCOL  *This,
  IN  UINT8                BarIndex,
  OUT UINT64               *Supports   OPTIONAL,
  OUT VOID                 **Resources OPTIONAL
  );

struct _EFI_PCI_IO_PROTOCOL {
  VOID                                    *PollMem;
  VOID                                    *PollIo;
  EFI_PCI_IO_PROTOCOL_ACCESS              Mem;
  EFI_PCI_IO_PROTOCOL_ACCESS              Io;
  EFI_PCI_IO_PROTOCOL_CONFIG_ACCESS       Pci;
  VOID                                    *CopyMem;
  EFI_PCI_IO_PROTOCOL_MAP                 Map;
  EFI_PCI_IO_PROTOCOL_UNMAP               Unmap;
  EFI_PCI_IO_PROTOCOL_ALLOCATE_BUFFER     AllocateBuffer;
  EFI_PCI_IO_PROTOCOL_FREE_BUFFER         FreeBuffer;
  VOID                                    *Flush;
  VOID                                    *GetLocation;
  EFI_PCI_IO_PROTOCOL_ATTRIBUTES          Attributes;
  EFI_PCI_IO_PROTOCOL_GET_BAR_ATTRIBUTES  GetBarAttributes;
  VOID                                    *SetBarAttributes;
  UINT64                                  RomSize;
  VOID                                    *RomImage;
};

//
// The QWORD address space descriptor (ACPI specification, 6.4.3.5.1), in
// which PCI I/O's GetBarAttributes describes a BAR. A memory BAR has ResType
// 0, its base in AddrRangeMin and its size in AddrLen.
//
#define ACPI_ADDRESS_SPACE_DESCRIPTOR  0x8A
#define ACPI_ADDRESS_SPACE_TYPE_MEM    0x00

#pragma pack(1)
typedef struct {
  UINT8     Desc;
  UINT16    Len;
  UINT8     ResType;
  UINT8     GenFlag;
  UINT8     SpecificFlag;
  UINT64    AddrSpaceGranularity;
  UINT64    AddrRangeMin;
  UINT64    AddrRangeMax;
  UINT64    AddrTranslationOffset;
  UINT64    AddrLen;
} EFI_ACPI_ADDRESS_SPACE_DESCRIPTOR;
#pragma pack()

//
// Graphics Output protocol (UEFI specification, 12.9).
//
#define EFI_GRAPHICS_OUTPUT_PROTOCOL_GUID \
  { 0x9042a9de, 0x23dc, 0x4a38, { 0x96, 0xfb, 0x7a, 0xde, 0xd0, 0x80, 0x51, 0x6a } }

typedef struct _EFI_GRAPHICS_OUTPUT_PROTOCOL EFI_GRAPHICS_OUTPUT_PROTOCOL;

typedef struct {
  UINT32    RedMask;
  UINT32    GreenMask;
  UINT32    BlueMask;
  UINT32    ReservedMask;
} EFI_PIXEL_BITMASK;

typedef enum {
  PixelRedGreenBlueReserved8BitPerColor,
  PixelBlueGreenRedReserved8BitPerColor,
  PixelBitMask,
  PixelBltOnly,
  PixelFormatMax
} EFI_GRAPHICS_PIXEL_FORMAT;

typedef struct {
  UINT32                       Version;
  UINT32                       HorizontalResolution;
  UINT32                       VerticalResolution;
  EFI_GRAPHICS_PIXEL_FORMAT    PixelFormat;
  EFI_PIXEL_BITMASK            PixelInformation;
  UINT32                       PixelsPerScanLine;
} EFI_GRAPHICS_OUTPUT_MODE_INFORMATION;

typedef struct {
  UINT32                                  MaxMode;
  UINT32                                  Mode;
  EFI_GRAPHICS_OUTPUT_MODE_INFORMATION    *Info;
  UINTN                                   SizeOfInfo;
  EFI_PHYSICAL_ADDRESS                    FrameBufferBase;
  UINTN                                   FrameBufferSize;
} EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE;

typedef struct {
  UINT8    Blue;
  UINT8    Green;
  UINT8    Red;
  UINT8    Reserved;
} EFI_GRAPHICS_OUTPUT_BLT_PIXEL;

typedef enum {
  EfiBltVideoFill,
  EfiBltVideoToBltBuffer,
  EfiBltBufferToVideo,
  EfiBltVideoToVideo,
  EfiGraphicsOutputBltOperationMax
} EFI_GRAPHICS_OUTPUT_BLT_OPERATION;

typedef
EFI_STATUS
(EFIAPI *EFI_GRAPHICS_OUTPUT_PROTOCOL_QUERY_MODE)(
  IN  EFI_GRAPHICS_OUTPUT_PROTOCOL          *This,
  IN  UINT32                                ModeNumber,
  OUT UINTN                                 *SizeOfInfo,
  OUT EFI_GRAPHICS_OUTPUT_MODE_INFORMATION  **Info
  );

typedef
EFI_STATUS
(EFIAPI *EFI_GRAPHICS_OUTPUT_PROTOCOL_SET_MODE)(
  IN  EFI_GRAPHICS_OUTPUT_PROTOCOL  *This,
  IN  UINT32                        ModeNumber
  );

typedef
EFI_STATUS
(EFIAPI *EFI_GRAPHICS_OUTPUT_PROTOCOL_BLT)(
  IN  EFI_GRAPHICS_OUTPUT_PROTOCOL       *This,
  IN  EFI_GRAPHICS_OUTPUT_BLT_PIXEL      *BltBuffer   OPTIONAL,
  IN  EFI_GRAPHICS_OUTPUT_BLT_OPERATION  BltOperation,
  IN  UINTN                              SourceX,
  IN  UINTN                              SourceY,
  IN  UINTN                              DestinationX,
  IN  UINTN                              DestinationY,
  IN  UINTN                              Width,
  IN  UINTN                              Height,
  IN  UINTN                              Delta         OPTIONAL
  );

struct _EFI_GRAPHICS_OUTPUT_PROTOCOL {
  EFI_GRAPHICS_OUTPUT_PROTOCOL_QUERY_MODE    QueryMode;
  EFI_GRAPHICS_OUTPUT_PROTOCOL_SET_MODE      SetMode;
  EFI_GRAPHICS_OUTPUT_PROTOCOL_BLT           Blt;
  EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE          *Mode;
};

//
// Protocol GUIDs, defined in ReimsVgpu.c.
//
extern EFI_GUID  gEfiDevicePathProtocolGuid;
extern EFI_GUID  gEfiDevicePathUtilitiesProtocolGuid;
extern EFI_GUID  gEfiDriverBindingProtocolGuid;
extern EFI_GUID  gEfiComponentNameProtocolGuid;
extern EFI_GUID  gEfiComponentName2ProtocolGuid;
extern EFI_GUID  gEfiDriverSupportedEfiVersionProtocolGuid;
extern EFI_GUID  gEfiPciIoProtocolGuid;
extern EFI_GUID  gEfiGraphicsOutputProtocolGuid;

#endif
