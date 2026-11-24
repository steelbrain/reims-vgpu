//! The window on Linux: SDL3's, on Wayland or X11, on the library's UI thread
//! (`super::sdl_thread`), and what only Linux has: the GPU a Wayland compositor composites on
//! (`feedback`), which a window's render thread asks for (`compositor_device`), and Linux's names
//! for the release chord's keys in the title ([`HOST_OS`]).

mod feedback;

pub(super) use feedback::main_device_of as compositor_device;

use crate::capture::HostOs;

/// The OS whose names the title gives the release chord's keys: Ctrl+Alt+Super, SDL's GUI key
/// being the Super key on Linux.
pub(super) const HOST_OS: HostOs = HostOs::Linux;
