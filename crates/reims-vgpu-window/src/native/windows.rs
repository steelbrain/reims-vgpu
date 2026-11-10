//! Windows' native SDL3 window and Vulkan swapchain.
//! SDL owns the HWND, input events and DPI conversion; the shared render loop creates
//! its Win32 Vulkan surface through SDL_Vulkan_CreateSurface on the presentation thread.
pub use super::sdl_vulkan::FrameSink;
pub(crate) use super::sdl_vulkan::{
    Surface, close, cursor_changed, guest_mode_changed, host_screens, now, open, sdl,
};
