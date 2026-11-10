//! Linux's window uses the shared SDL/Vulkan implementation.
pub use super::sdl_vulkan::FrameSink;
pub(crate) use super::sdl_vulkan::{
    Surface, close, cursor_changed, guest_mode_changed, host_screens, now, open, sdl,
};
