//! The window on Linux: SDL3's, on Wayland or X11, on the library's UI thread
//! (`super::sdl_thread`), and what only Linux has: the GPU a Wayland compositor composites on
//! (`feedback`), which a window's render thread asks for (`compositor_device`).

mod feedback;

pub(super) use feedback::main_device_of as compositor_device;
