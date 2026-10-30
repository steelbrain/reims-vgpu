/** @file
  Graphics Output Protocol of the reims-graphics device: one mode, a linear
  32-bit BGRX framebuffer in the region at RGFX_ALLOCATIONS_OFFSET, whose rows
  are the private data's Stride bytes apart.
**/

#include "ReimsGraphics.h"

STATIC
UINT8 *
FrameBuffer (
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private
  )
{
  return (UINT8 *)(UINTN)(Private->RegionBase + RGFX_ALLOCATIONS_OFFSET);
}

STATIC
VOID
SetModeInformation (
  IN  REIMS_GRAPHICS_PRIVATE_DATA           *Private,
  OUT EFI_GRAPHICS_OUTPUT_MODE_INFORMATION  *Info
  )
{
  Info->Version                       = 0;
  Info->HorizontalResolution          = Private->Width;
  Info->VerticalResolution            = Private->Height;
  Info->PixelFormat                   = PixelBlueGreenRedReserved8BitPerColor;
  Info->PixelInformation.RedMask      = 0;
  Info->PixelInformation.GreenMask    = 0;
  Info->PixelInformation.BlueMask     = 0;
  Info->PixelInformation.ReservedMask = 0;
  Info->PixelsPerScanLine             = Private->Stride / 4;
}

EFI_STATUS
EFIAPI
ReimsGraphicsGraphicsOutputQueryMode (
  IN  EFI_GRAPHICS_OUTPUT_PROTOCOL          *This,
  IN  UINT32                                ModeNumber,
  OUT UINTN                                 *SizeOfInfo,
  OUT EFI_GRAPHICS_OUTPUT_MODE_INFORMATION  **Info
  )
{
  if ((Info == NULL) || (SizeOfInfo == NULL) || (ModeNumber >= This->Mode->MaxMode)) {
    return EFI_INVALID_PARAMETER;
  }

  if (EFI_ERROR (gBS->AllocatePool (EfiBootServicesData, sizeof (**Info), (VOID **)Info))) {
    return EFI_OUT_OF_RESOURCES;
  }

  *SizeOfInfo = sizeof (**Info);
  SetModeInformation (REIMS_GRAPHICS_PRIVATE_DATA_FROM_GRAPHICS_OUTPUT_THIS (This), *Info);
  return EFI_SUCCESS;
}

/**
  Clear the framebuffer to black and point the display at it.
**/
EFI_STATUS
EFIAPI
ReimsGraphicsGraphicsOutputSetMode (
  IN  EFI_GRAPHICS_OUTPUT_PROTOCOL  *This,
  IN  UINT32                        ModeNumber
  )
{
  REIMS_GRAPHICS_PRIVATE_DATA  *Private;
  volatile UINT64              *Buffer;
  UINTN                        Index;
  UINTN                        Size;

  Private = REIMS_GRAPHICS_PRIVATE_DATA_FROM_GRAPHICS_OUTPUT_THIS (This);
  if (ModeNumber >= This->Mode->MaxMode) {
    return EFI_UNSUPPORTED;
  }

  Size   = (UINTN)Private->Stride * Private->Height;
  Buffer = (volatile UINT64 *)FrameBuffer (Private);
  for (Index = 0; Index < Size / sizeof (UINT64); Index++) {
    Buffer[Index] = 0;
  }

  if (EFI_ERROR (ReimsGraphicsSetScanout (Private, FALSE))) {
    return EFI_DEVICE_ERROR;
  }

  This->Mode->Mode            = ModeNumber;
  This->Mode->SizeOfInfo      = sizeof (EFI_GRAPHICS_OUTPUT_MODE_INFORMATION);
  This->Mode->FrameBufferBase = Private->RegionBase + RGFX_ALLOCATIONS_OFFSET;
  This->Mode->FrameBufferSize = Size;
  SetModeInformation (Private, This->Mode->Info);
  return EFI_SUCCESS;
}

/**
  Whether a Width x Height rectangle at X, Y lies inside a Limit x Height
  area.
**/
STATIC
BOOLEAN
RectInside (
  IN UINTN  X,
  IN UINTN  Y,
  IN UINTN  Width,
  IN UINTN  Height,
  IN UINTN  LimitWidth,
  IN UINTN  LimitHeight
  )
{
  return (X <= LimitWidth) && (Width <= LimitWidth - X) &&
         (Y <= LimitHeight) && (Height <= LimitHeight - Y);
}

/**
  Copy Count pixels, backward when the destination lies after the source, so
  overlapping runs within one row copy as a move does.
**/
STATIC
VOID
CopyPixels (
  OUT UINT32        *Destination,
  IN  CONST UINT32  *Source,
  IN  UINTN         Count
  )
{
  UINTN  Index;

  if (Destination > Source) {
    for (Index = Count; Index > 0; Index--) {
      Destination[Index - 1] = Source[Index - 1];
    }
  } else {
    for (Index = 0; Index < Count; Index++) {
      Destination[Index] = Source[Index];
    }
  }
}

EFI_STATUS
EFIAPI
ReimsGraphicsGraphicsOutputBlt (
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
  )
{
  REIMS_GRAPHICS_PRIVATE_DATA  *Private;
  UINT8                        *Frame;
  UINT8                        *Buffer;
  UINTN                        Pitch;
  UINTN                        Row;
  UINTN                        Y;
  UINTN                        X;
  UINT32                       Fill;
  UINT32                       *Line;

  Private = REIMS_GRAPHICS_PRIVATE_DATA_FROM_GRAPHICS_OUTPUT_THIS (This);
  Frame   = FrameBuffer (Private);
  Buffer  = (UINT8 *)BltBuffer;
  Pitch   = Private->Stride;
  if ((Width == 0) || (Height == 0) || ((BltOperation != EfiBltVideoToVideo) && (BltBuffer == NULL))) {
    return EFI_INVALID_PARAMETER;
  }

  if (Delta == 0) {
    Delta = Width * sizeof (EFI_GRAPHICS_OUTPUT_BLT_PIXEL);
  }

  //
  // The rectangle read from the screen, then the one written to it.
  //
  if (((BltOperation == EfiBltVideoToBltBuffer) || (BltOperation == EfiBltVideoToVideo)) &&
      !RectInside (SourceX, SourceY, Width, Height, Private->Width, Private->Height))
  {
    return EFI_INVALID_PARAMETER;
  }

  if ((BltOperation != EfiBltVideoToBltBuffer) &&
      !RectInside (DestinationX, DestinationY, Width, Height, Private->Width, Private->Height))
  {
    return EFI_INVALID_PARAMETER;
  }

  switch (BltOperation) {
    case EfiBltVideoFill:
      Fill = *(UINT32 *)BltBuffer;
      for (Y = 0; Y < Height; Y++) {
        Line = (UINT32 *)(Frame + (DestinationY + Y) * Pitch) + DestinationX;
        for (X = 0; X < Width; X++) {
          Line[X] = Fill;
        }
      }

      break;

    case EfiBltVideoToBltBuffer:
      for (Y = 0; Y < Height; Y++) {
        CopyPixels (
          (UINT32 *)(Buffer + (DestinationY + Y) * Delta) + DestinationX,
          (UINT32 *)(Frame + (SourceY + Y) * Pitch) + SourceX,
          Width
          );
      }

      break;

    case EfiBltBufferToVideo:
      for (Y = 0; Y < Height; Y++) {
        CopyPixels (
          (UINT32 *)(Frame + (DestinationY + Y) * Pitch) + DestinationX,
          (UINT32 *)(Buffer + (SourceY + Y) * Delta) + SourceX,
          Width
          );
      }

      break;

    case EfiBltVideoToVideo:
      //
      // Rows go bottom up when the destination lies below the source.
      //
      for (Y = 0; Y < Height; Y++) {
        Row = (DestinationY > SourceY) ? Height - 1 - Y : Y;
        CopyPixels (
          (UINT32 *)(Frame + (DestinationY + Row) * Pitch) + DestinationX,
          (UINT32 *)(Frame + (SourceY + Row) * Pitch) + SourceX,
          Width
          );
      }

      break;

    default:
      return EFI_INVALID_PARAMETER;
  }

  return EFI_SUCCESS;
}

VOID
ReimsGraphicsGraphicsOutputInit (
  IN REIMS_GRAPHICS_PRIVATE_DATA  *Private
  )
{
  EFI_GRAPHICS_OUTPUT_PROTOCOL  *GraphicsOutput;

  GraphicsOutput                   = &Private->GraphicsOutput;
  GraphicsOutput->QueryMode        = ReimsGraphicsGraphicsOutputQueryMode;
  GraphicsOutput->SetMode          = ReimsGraphicsGraphicsOutputSetMode;
  GraphicsOutput->Blt              = ReimsGraphicsGraphicsOutputBlt;
  GraphicsOutput->Mode             = &Private->GraphicsOutputMode;
  GraphicsOutput->Mode->MaxMode    = 1;
  GraphicsOutput->Mode->Mode       = 0;
  GraphicsOutput->Mode->Info       = &Private->ModeInformation;
  GraphicsOutput->Mode->SizeOfInfo = sizeof (EFI_GRAPHICS_OUTPUT_MODE_INFORMATION);
  SetModeInformation (Private, GraphicsOutput->Mode->Info);
}
