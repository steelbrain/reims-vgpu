//! A Vulkan surface of a window's `CAMetalLayer` ([`LayerTarget`]), which the display's sink makes
//! its swapchain over (`crate::present`): on the render thread, with `VK_EXT_metal_surface`, on
//! the instance the sink gives.
//!
//! AppKit's window and SDL's alike: each window has its layer already, AppKit's view's or SDL's
//! Metal view's (`sdl::attach_layer`), which the render thread holds, so one target serves both.
//! SDL's own `SDL_Vulkan_CreateSurface` is not used for SDL's window: on macOS it makes another
//! Metal view in the window at each call (SDL 3.4.18, `SDL_cocoavulkan.m`, which calls
//! `Cocoa_Metal_CreateView` before `vkCreateMetalSurfaceEXT`), on top of the one the window
//! draws into, and it needs SDL to open a Vulkan library of its own (`SDL_WINDOW_VULKAN`), which
//! `dlopen`'s default search does not find where Homebrew installs it, as the runtime's search
//! does.
//!
//! **The layer off the main thread.** The surface's driver retains the layer, and its swapchains
//! set the layer up from the render thread (its device, pixel format, drawable size and count,
//! colour space, opacity), as Core Animation lets any thread do. The surface goes before the
//! render thread lets its own reference to the layer go to the main thread (`render`), and the
//! window's view holds another: the driver's release is never the layer's last.

use std::ffi::c_void;

use ash::vk;

use super::render::LayerRef;
use crate::present::VulkanTarget;

/// A window's layer, as a Vulkan surface is made of it.
pub(crate) struct LayerTarget<'a>(pub(crate) &'a LayerRef);

impl VulkanTarget for LayerTarget<'_> {
    /// A surface of the layer (`vkCreateMetalSurfaceEXT`, which `get_instance_proc_addr` looks up
    /// on `instance`): `None` where the instance has no such command (its loader lists no
    /// `VK_EXT_metal_surface`) or makes none, which is logged.
    fn create_surface(
        &self,
        get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
        instance: vk::Instance,
    ) -> Option<vk::SurfaceKHR> {
        // SAFETY: a live instance's look-up of an instance command by its name.
        let found =
            unsafe { get_instance_proc_addr(instance, c"vkCreateMetalSurfaceEXT".as_ptr()) };
        let Some(found) = found else {
            log::warn!(
                target: "reims_vgpu_window",
                "the instance has no vkCreateMetalSurfaceEXT: the window has no Vulkan surface"
            );
            return None;
        };
        // SAFETY: the command the loader answers for the name, whose signature is
        // `vkCreateMetalSurfaceEXT`'s.
        let create: vk::PFN_vkCreateMetalSurfaceEXT = unsafe { std::mem::transmute(found) };
        let layer: *const c_void = self.0.as_ptr();
        let info = vk::MetalSurfaceCreateInfoEXT::default().layer(layer);
        let mut surface = vk::SurfaceKHR::null();
        // SAFETY: a valid create info naming the window's live layer, which outlives the
        // surface (the module doc), on a live instance.
        let made = unsafe { create(instance, &info, std::ptr::null(), &mut surface) };
        if made == vk::Result::SUCCESS {
            Some(surface)
        } else {
            log::warn!(
                target: "reims_vgpu_window",
                "vkCreateMetalSurfaceEXT: {made}: the window has no Vulkan surface"
            );
            None
        }
    }
}
