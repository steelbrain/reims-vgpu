/** @file
  Ambrosia's graphics driver interface, revision 1, in C: what Ambrosia, the
  boot firmware of arm64 macOS guests on QEMU's vmapple machine, gives a
  graphics driver module, and what the module gives back. Ambrosia's
  notes/graphics-driver.md is the interface's guide, and its
  crates/ambrosia/src/graphics.rs the types this file declares again.

  A module is an ELF64 aarch64 static PIE whose relocations are all
  R_AARCH64_RELATIVE. Ambrosia loads it from its configuration bundle, calls
  its entry point once, on the boot CPU at EL1 with the MMU on and an identity
  map, and never calls into it again. The rest of the first 4 GiB, the ECAM
  and the PCIe window among it, is mapped as device memory: each access there
  is aligned to its size, with no exclusive or atomic access.
**/

#ifndef AMBROSIA_GRAPHICS_H_
#define AMBROSIA_GRAPHICS_H_

#include <stddef.h>
#include <stdint.h>

//
// The revision AMBROSIA_SERVICES.Revision carries. A later revision only adds
// fields at the end of AMBROSIA_SERVICES.
//
#define AMBROSIA_GRAPHICS_REVISION  1

typedef struct AMBROSIA_SERVICES AMBROSIA_SERVICES;

/**
  Write Length bytes of UTF-8 at Text as one line of Ambrosia's log, to the
  UART and, once a framebuffer is up, to the screen. Ambrosia adds the line's
  prefix and its end.
**/
typedef
void
(*AMBROSIA_LOG)(
  const AMBROSIA_SERVICES  *Services,
  const uint8_t            *Text,
  size_t                   Length
  );

//
// What Ambrosia gives the module. Addresses are physical, and are the
// module's pointers.
//
struct AMBROSIA_SERVICES {
  uint32_t        Revision;      // AMBROSIA_GRAPHICS_REVISION or later
  uint32_t        Reserved;      // 0
  //
  // The PCIe ECAM: a function's configuration space is 4 KiB at
  // EcamBase + (Bus << 20) + (Device << 15) + (Function << 12).
  //
  uint64_t        EcamBase;
  uint64_t        EcamSize;
  //
  // The PCIe memory window, below 4 GiB, from which the module gives its
  // device's BARs addresses, which the kernel keeps. The kernel gives every
  // other BAR an address from the same window.
  //
  uint64_t        WindowBase;
  uint64_t        WindowSize;
  AMBROSIA_LOG    Log;
};

//
// The display the module set up, which it fills when it returns 0. Ambrosia
// zeroes it before the call. Its pixels are 32 bits, 0x00RRGGBB
// little-endian: blue is the first byte in memory.
//
typedef struct {
  uint64_t    Base;       // physical, in one of the device's BARs, in the window
  uint32_t    Width;      // pixels
  uint32_t    Height;     // rows
  uint32_t    Stride;     // bytes from one row to the next: a multiple of 4, at least 4 * Width
  uint32_t    Reserved;   // 0
} AMBROSIA_FRAMEBUFFER;

/**
  The module's entry point, its ELF header's e_entry. It returns 0 when it
  drives a display and has filled Framebuffer, and anything else when it found
  no device it drives, or failed; Ambrosia then boots with no console but the
  UART, and logs the value.
**/
typedef
int32_t
(*AMBROSIA_GRAPHICS_ENTRY)(
  const AMBROSIA_SERVICES  *Services,
  AMBROSIA_FRAMEBUFFER     *Framebuffer
  );

//
// The interface's layout.
//
_Static_assert (__builtin_offsetof (AMBROSIA_SERVICES, EcamBase) == 8, "EcamBase");
_Static_assert (__builtin_offsetof (AMBROSIA_SERVICES, WindowBase) == 24, "WindowBase");
_Static_assert (__builtin_offsetof (AMBROSIA_SERVICES, Log) == 40, "Log");
_Static_assert (sizeof (AMBROSIA_SERVICES) == 48, "AMBROSIA_SERVICES");
_Static_assert (__builtin_offsetof (AMBROSIA_FRAMEBUFFER, Width) == 8, "Width");
_Static_assert (__builtin_offsetof (AMBROSIA_FRAMEBUFFER, Stride) == 16, "Stride");
_Static_assert (sizeof (AMBROSIA_FRAMEBUFFER) == 24, "AMBROSIA_FRAMEBUFFER");

#endif
