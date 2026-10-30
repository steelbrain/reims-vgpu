//! What Linux's window asks of the graphics display's Vulkan sink ([`FrameSink`]): a presenter
//! over the window's swapchain, which SDL's window makes its surface for.

use reims_vgpu_window::{FrameSink, Presenter, VulkanTarget};

use super::VulkanScanoutSink;

impl FrameSink for VulkanScanoutSink {
    /// The presenter ([`VulkanScanoutSink::make_presenter`]).
    fn presenter(&self, target: &dyn VulkanTarget) -> Option<Box<dyn Presenter>> {
        self.make_presenter(target)
    }
}
