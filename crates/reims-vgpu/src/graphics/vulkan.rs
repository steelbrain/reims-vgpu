//! A graphics display's window sink on the Vulkan platform, on either OS: [`VulkanScanoutSink`]
//! makes a presenter over a swapchain of the window's surface, and each draw blits the scanout
//! into the swapchain's next image (its usage takes transfers) and presents it.

use std::sync::Arc;

use reims_vgpu_core::platform::command::CommandBuffer as _;
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_graphics::region::Region;
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::swapchain::{Acquired, Swapchain};
use reims_vgpu_window::{Drawn, DrawnCursor, FrameSink, PixelRect, Presenter, VulkanTarget};

use super::sink::ScanoutDraw;
use super::{Presented, Shown};
use crate::entry;
use crate::entry::no_catch;

/// The sink of a graphics display on the Vulkan platform, over `region`; `None` where the
/// runtime refuses a queue.
pub(super) fn sink(
    platform: Arc<VulkanPlatform>,
    region: &Arc<Region>,
    shown: &Arc<Shown>,
    presented: &Arc<Presented<VulkanPlatform>>,
) -> Option<Arc<dyn FrameSink>> {
    let draw = ScanoutDraw::new(platform, region, shown, presented)?;
    Some(Arc::new(VulkanScanoutSink {
        draw: Arc::new(draw),
    }))
}

/// What presents a graphics display's scanout in its window on the Vulkan platform.
pub(super) struct VulkanScanoutSink {
    /// The scanout's buffer, the queue and the scanout; shared with the presenters.
    draw: Arc<ScanoutDraw<VulkanPlatform>>,
}

impl FrameSink for VulkanScanoutSink {
    /// The window's surface on the runtime's instance and a swapchain over it, on the sink's
    /// queue's device; none where either is refused, which is logged.
    fn presenter(&self, target: &dyn VulkanTarget) -> Option<Box<dyn Presenter>> {
        // SAFETY: the target makes the surface on the instance it is given, and hands it over.
        let made = entry::run(|_| unsafe {
            Swapchain::over(&self.draw.queue, |get_instance_proc_addr, instance| {
                target.create_surface(get_instance_proc_addr, instance)
            })
        });
        match made {
            Ok(swapchain) => Some(Box::new(ScanoutPresenter {
                draw: Arc::clone(&self.draw),
                swapchain,
            })),
            Err(refused) => {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "no swapchain for the graphics display's window ({refused}): it stays empty"
                );
                None
            }
        }
    }
}

/// A graphics window's presenter: the scanout's draw and the window's swapchain, on its render
/// thread.
struct ScanoutPresenter {
    /// The scanout's buffer, the queue and the scanout.
    draw: Arc<ScanoutDraw<VulkanPlatform>>,
    /// The swapchain.
    swapchain: Swapchain,
}

impl Presenter for ScanoutPresenter {
    /// The next image, the scanout blitted into it, and its present once the runtime has
    /// scheduled the blit; nothing, the image kept for the next draw, where the draw is refused
    /// (`sink`'s module doc, "A refused draw"). `cursor` is never given (`graphics`'s module doc).
    fn draw(
        &mut self,
        drawable: (u64, u64),
        region: PixelRect,
        _cursor: Option<&DrawnCursor>,
    ) -> Drawn {
        entry::run(|_| {
            self.draw.drawn(drawable);
            let frame = match self.swapchain.acquire(drawable) {
                Acquired::Image(frame) => frame,
                Acquired::Retry => return Drawn::Retry,
                Acquired::Failed => return Drawn::Ended,
            };
            let Some(texture) = self.swapchain.texture(frame) else {
                return Drawn::Nothing;
            };
            let swapchain = &self.swapchain;
            let committed = self.draw.scoped(|| {
                let (command_buffer, presents) = self.draw.encode(texture, region)?;
                // A draw dropped here answers its presents at the next refresh.
                if !swapchain.encode_ready(frame, &command_buffer) {
                    return None;
                }
                presents.after(&command_buffer);
                no_catch(command_buffer.commit(OnFault::Die));
                Some(command_buffer)
            });
            let Some(command_buffer) = committed else {
                return Drawn::Nothing;
            };
            self.swapchain.present(frame, command_buffer);
            Drawn::Presented
        })
    }
}
