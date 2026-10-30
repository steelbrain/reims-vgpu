/** @file
  Component Name and Component Name 2 protocols of the reims-graphics
  device's driver: English names, "eng" for the first and "en" for the
  second.
**/

#include "ReimsGraphics.h"

STATIC CHAR16  mDriverName[]     = L"Reims Graphics Driver";
STATIC CHAR16  mControllerName[] = L"Reims Graphics";

STATIC
BOOLEAN
SameString (
  IN CONST CHAR8  *First,
  IN CONST CHAR8  *Second
  )
{
  while ((*First != '\0') && (*First == *Second)) {
    First++;
    Second++;
  }

  return *First == *Second;
}

EFI_STATUS
EFIAPI
ReimsGraphicsGetDriverName (
  IN  EFI_COMPONENT_NAME_PROTOCOL  *This,
  IN  CHAR8                        *Language,
  OUT CHAR16                       **DriverName
  )
{
  if ((Language == NULL) || (DriverName == NULL)) {
    return EFI_INVALID_PARAMETER;
  }

  if (!SameString (Language, This->SupportedLanguages)) {
    return EFI_UNSUPPORTED;
  }

  *DriverName = mDriverName;
  return EFI_SUCCESS;
}

/**
  The controller's name, for a controller this driver manages; its child has
  none.
**/
EFI_STATUS
EFIAPI
ReimsGraphicsGetControllerName (
  IN  EFI_COMPONENT_NAME_PROTOCOL  *This,
  IN  EFI_HANDLE                   ControllerHandle,
  IN  EFI_HANDLE                   ChildHandle        OPTIONAL,
  IN  CHAR8                        *Language,
  OUT CHAR16                       **ControllerName
  )
{
  EFI_STATUS  Status;
  VOID        *PciIo;

  if ((Language == NULL) || (ControllerName == NULL)) {
    return EFI_INVALID_PARAMETER;
  }

  if (ChildHandle != NULL) {
    return EFI_UNSUPPORTED;
  }

  //
  // The driver manages the controller when its PCI I/O is already open by
  // the driver.
  //
  Status = gBS->OpenProtocol (
                  ControllerHandle,
                  &gEfiPciIoProtocolGuid,
                  &PciIo,
                  gReimsGraphicsDriverBinding.DriverBindingHandle,
                  ControllerHandle,
                  EFI_OPEN_PROTOCOL_BY_DRIVER
                  );
  if (!EFI_ERROR (Status)) {
    gBS->CloseProtocol (
           ControllerHandle,
           &gEfiPciIoProtocolGuid,
           gReimsGraphicsDriverBinding.DriverBindingHandle,
           ControllerHandle
           );
    return EFI_UNSUPPORTED;
  }

  if (Status != EFI_ALREADY_STARTED) {
    return EFI_UNSUPPORTED;
  }

  if (!SameString (Language, This->SupportedLanguages)) {
    return EFI_UNSUPPORTED;
  }

  *ControllerName = mControllerName;
  return EFI_SUCCESS;
}

EFI_COMPONENT_NAME_PROTOCOL  gReimsGraphicsComponentName = {
  ReimsGraphicsGetDriverName,
  ReimsGraphicsGetControllerName,
  "eng"
};

//
// The second protocol's functions take the same arguments.
//
EFI_COMPONENT_NAME2_PROTOCOL  gReimsGraphicsComponentName2 = {
  (EFI_COMPONENT_NAME2_GET_DRIVER_NAME)ReimsGraphicsGetDriverName,
  (EFI_COMPONENT_NAME2_GET_CONTROLLER_NAME)ReimsGraphicsGetControllerName,
  "en"
};
