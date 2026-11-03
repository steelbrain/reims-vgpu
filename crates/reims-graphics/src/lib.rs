//! The reims-graphics device (PCI 1234:5247): a GPU device of our own beside the Apple paravirt
//! lane, for x86-64 macOS guests, with our own kext, protocol and QEMU device
//! (`docs/graphics-kext/design.md`). Stage 2 has the transport and an unaccelerated display.
//!
//! - [`wire`] is the format the guest and the host share: the registers, the region's layout,
//!   the rings and the control commands and events. cbindgen writes it into
//!   `include/reims_graphics_wire.h`, which the guest's kext and option ROM include.
//! - [`ops`] is the command-buffer op stream in that format: the `COMMAND_BUFFER` message and the
//!   encoder ops (begin/end, and the render, compute and blit groups) the guest serialises.
//! - [`icb`] is the records of an `ICB_WRITE`: the CPU's encodes into an indirect command
//!   buffer, in that format too.
//! - [`Device`] is the host side, as the host's process runs it: the region (BAR 2), the
//!   registers (BAR 0), the control ring's thread and the display's refresh thread, over the
//!   host's callbacks ([`Host`]) and what shows the display ([`Screen`]). `reims-vgpu` shows it
//!   in a window of the library's, and the C API hands it to QEMU's `reims-graphics-pci`.
//! - `ring` is the host's two ends of a ring, `control` the commands, [`display`] the display's
//!   modes, scanout and cursor, and [`region`] the region's memory.

mod control;
pub mod device;
pub mod display;
pub mod icb;
pub mod managed;
pub mod ops;
pub mod region;
mod ring;
mod wake;
pub mod wire;

pub use device::{CompleteAnswer, CompletionPoster, Device, Host, PresentAnswer, REGION_SIZE};
pub use display::{CursorImage, FixedScreen, Mode, Scanout, Screen};
