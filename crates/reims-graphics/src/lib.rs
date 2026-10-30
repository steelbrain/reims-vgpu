//! The reims-graphics device (PCI 1234:5247): a GPU device of our own beside the Apple paravirt
//! lane, for x86-64 macOS guests, with our own kext, protocol and QEMU device
//! (`docs/graphics-kext/design.md`).
//!
//! - [`wire`] is the format the guest and the host share: the registers, the region's layout,
//!   the rings and the control commands and events. cbindgen writes it into
//!   `include/reims_graphics_wire.h`, which the guest's kext and option ROM include.

pub mod wire;
