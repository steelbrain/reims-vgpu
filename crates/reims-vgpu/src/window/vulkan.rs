//! What draws a window display's frames on the Vulkan platform: the runtime draws each frame into
//! the window's swapchain, on either OS (`swapchain.rs`), and each OS's window asks the sink for
//! it its own way (`linux.rs`, `macos.rs`).

#[cfg(any(target_os = "linux", target_os = "windows"))]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
mod swapchain;

pub(crate) use swapchain::sink;
