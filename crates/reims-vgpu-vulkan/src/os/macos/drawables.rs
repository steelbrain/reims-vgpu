//! A window's swapchain image's drawable, fetched on the window's render thread, where the driver
//! is MoltenVK ([`Drawables`]).
//!
//! **Why ahead of the frame.** MoltenVK (1.4.1) takes a swapchain image's `CAMetalDrawable` from
//! the layer (`nextDrawable`) lazily, at the first use of the image's Metal texture after the
//! image was acquired (`MVKPresentableSwapchainImage::getCAMetalDrawable`): the encode of the
//! frame drawn into it, or its present. It encodes a command buffer as it is submitted, on the
//! thread that submits it (`MVK_CONFIG_SYNCHRONOUS_QUEUE_SUBMITS`, on by default, and the runtime
//! leaves it so), and the runtime submits every queue's batches under its device's one scheduler
//! lock, on the committing thread or on the device's scheduler thread. `nextDrawable` blocks, up
//! to a second, while every drawable of the layer (`maximumDrawableCount`, which MoltenVK sets to
//! the swapchain's image count) is with the compositor, and that is apart from the image's
//! acquire fence: the fence signals once the image's last present is done, while a drawable comes
//! back to the layer only once the compositor has let it go. A drawable fetched in the frame's
//! submission would so hold every queue's submissions, the guest's among them, for as long as it
//! waits. Presenting the frame on the window's own queue would not help: the encode is still the
//! submission's, under the same lock.
//!
//! **How.** The render thread fetches it as it takes the image, through `VK_EXT_metal_objects`'
//! export of the image's Metal texture (`vkExportMetalObjectsEXT`), which MoltenVK answers with
//! the drawable's texture, fetching the drawable (`MVKDevice::getMetalObjects`). A wait, where
//! there is one, is then the render thread's, as `nextDrawable` is on a Metal window, and the
//! frame's encode finds the drawable fetched. MoltenVK lets go of an image's
//! drawable as the image is acquired again (`acquireAndSignalWhenAvailable`), so the fetch
//! follows each acquire, never precedes it.
//!
//! **Only MoltenVK.** The export is MoltenVK's way to reach the drawable, and another driver's
//! image has no drawable to fetch; Linux fetches nothing (`crate::os`'s `Drawables` there). On a
//! device of another driver, or one without the extension, nothing is fetched ahead.

use ash::vk;
use metal2vulkan::vk::DriverId;
use metal2vulkan_runtime as rt;

/// `VK_EXT_metal_objects`' export on a MoltenVK device, which fetches a swapchain image's
/// drawable (the module doc).
#[derive(Clone)]
pub(crate) struct Drawables {
    /// The device's handle.
    device: vk::Device,
    /// `vkExportMetalObjectsEXT`, the driver's own, which reads no loader word.
    export: vk::PFN_vkExportMetalObjectsEXT,
}

impl std::fmt::Debug for Drawables {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Drawables").finish_non_exhaustive()
    }
}

impl Drawables {
    /// The export on `raw`, looked up as a swapchain is made, through the runtime's look-up
    /// (`rt::Device::device_proc_addr`, `crate::swapchain`): `None` where its driver is not
    /// MoltenVK, or the device options ask for no `VK_EXT_metal_objects` (a loader or device
    /// that lists no window extensions, `crate::os`'s `wsi`).
    pub(crate) fn load(raw: &rt::Device) -> Option<Drawables> {
        if raw.properties().description.driver_id != Some(DriverId::MOLTENVK) {
            return None;
        }
        let asked = crate::platform::device_options()
            .device_extensions
            .iter()
            .any(|name| name.as_c_str() == ash::ext::metal_objects::NAME);
        if !asked {
            return None;
        }
        let found = raw.device_proc_addr(c"vkExportMetalObjectsEXT")?;
        // SAFETY: the command the device answers for the name, whose signature is
        // `vkExportMetalObjectsEXT`'s.
        let export: vk::PFN_vkExportMetalObjectsEXT = unsafe { std::mem::transmute(found) };
        // SAFETY: the handle is the device's, read while it lives; the swapchain that fetches
        // holds the device.
        let device = unsafe { raw.raw() }.device.handle();
        Some(Drawables { device, export })
    }

    /// Fetches the drawable of `image`, a swapchain image just acquired, its fence waited for:
    /// whether it has one, which the frame then draws into. False where the layer gave none in
    /// its time (MoltenVK asks it as many times as the swapchain has images, each up to a second):
    /// the draw is tried again, and the fetch with it.
    ///
    /// # Safety
    ///
    /// `image` is an image of a live swapchain of the device, acquired, and no other thread uses
    /// it meanwhile.
    pub(crate) unsafe fn fetch(&self, image: vk::Image) -> bool {
        let mut texture = vk::ExportMetalTextureInfoEXT::default()
            .image(image)
            .plane(vk::ImageAspectFlags::PLANE_0);
        {
            let mut info = vk::ExportMetalObjectsInfoEXT::default().push_next(&mut texture);
            // SAFETY: the caller's; the info chains one texture export of a live image.
            unsafe { (self.export)(self.device, &mut info) };
        }
        !texture.mtl_texture.is_null()
    }
}
