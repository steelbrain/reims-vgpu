//! What macOS's window asks of the graphics display's Vulkan sink ([`FrameSink`]): it presents
//! itself, through a presenter over a swapchain of the window's `CAMetalLayer`, as the Vulkan
//! platform's display sink does (`crate::window`'s `vulkan/macos.rs`).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use objc2_quartz_core::CAMetalDrawable;
use reims_vgpu_window::{DrawnCursor, FrameSink, PixelRect, Presenter, VulkanTarget};

use super::VulkanScanoutSink;

impl FrameSink for VulkanScanoutSink {
    /// None: the swapchain sets the layer's device.
    fn device(&self) -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
        None
    }

    /// Nothing: the sink presents itself, and is given no drawable.
    fn draw(
        &self,
        _drawable: &ProtocolObject<dyn CAMetalDrawable>,
        _region: PixelRect,
        _cursor: Option<&DrawnCursor>,
    ) -> bool {
        false
    }

    /// False: the swapchain's images take transfers.
    fn render_target_only(&self) -> bool {
        false
    }

    /// True: the sink presents through [`FrameSink::presenter`].
    fn presents(&self) -> bool {
        true
    }

    /// The presenter ([`VulkanScanoutSink::make_presenter`]).
    fn presenter(&self, target: &dyn VulkanTarget) -> Option<Box<dyn Presenter>> {
        self.make_presenter(target)
    }
}
