//! A graphics display's window sink on the rt platform: [`RtScanoutSink`] makes a presenter over a
//! swapchain of the window's surface on rt's device ([`RtSwapchain`], rt's drawables), and each
//! draw draws the display into the swapchain's next image (`super::sink`), and presents it.
//!
//! **The compositor's GPU.** The window presents from rt's device, the process's one device,
//! wherever the host's compositor composites it. Where that is another GPU (the window system
//! names it: [`VulkanTarget::compositor_device`]), each image presented crosses to it through the
//! window system (an RTX 3090 on PCIe 3.0 x4 spends 16.4 ms of its link on a 3648 x 2039
//! present, the module doc of `reims_vgpu_vulkan::swapchain`). No window device is made on the
//! compositor's GPU: rt makes one device for everything. The route is logged once a sink, at its
//! first presenter.
//!
//! **A frame's GPU time** is counted for the host's summary where it counts the window's times
//! (`reims_vgpu_vulkan::rt_platform::window_times`): a completed handler of the frame's command
//! buffer, added after its presents' handler and before its commit, hands on its GPU start and
//! end.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reims_vgpu_graphics::region::Region;
use reims_vgpu_vulkan::RtPlatform;
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CompletedInfo};
use reims_vgpu_vulkan::core::raise::OnFault;
use reims_vgpu_vulkan::rt_platform::swapchain::RtSwapchain;
use reims_vgpu_vulkan::rt_platform::window_times;
use reims_vgpu_vulkan::swapchain::{Acquired, Refused};
use reims_vgpu_window::{Drawn, DrawnCursor, FrameSink, PixelRect, Presenter, VulkanTarget};

use super::sink::ScanoutDraw;
use super::{Presented, Shown};
use crate::entry;
use crate::entry::no_catch;

/// The sink of a graphics display on the rt platform, over `region`; `None` where rt refuses a
/// queue.
pub(super) fn sink(
    platform: Arc<RtPlatform>,
    region: &Arc<Region>,
    shown: &Arc<Shown>,
    presented: &Arc<Presented<RtPlatform>>,
) -> Option<Arc<dyn FrameSink>> {
    let draw = ScanoutDraw::new(platform, region, shown, presented)?;
    Some(Arc::new(RtScanoutSink {
        draw: Arc::new(draw),
        routed: AtomicBool::new(false),
    }))
}

/// What presents a graphics display's scanout in its window on the rt platform.
pub(super) struct RtScanoutSink {
    /// The scanout's buffer, the queue and the scanout on rt's device; shared with the
    /// presenters.
    draw: Arc<ScanoutDraw<RtPlatform>>,
    /// The route was logged (the module doc).
    routed: AtomicBool,
}

impl RtScanoutSink {
    /// Logs where the window is composited, against rt's device, once (the module doc).
    fn log_route(&self, target: &dyn VulkanTarget) {
        if self.routed.swap(true, Ordering::AcqRel) {
            return;
        }
        let Some(node) = target.compositor_device() else {
            log::info!(
                target: "reims_vgpu::graphics",
                "the window system names no GPU the window is composited on: the window presents \
                 from the runtime's GPU"
            );
            return;
        };
        let (major, minor) = node;
        match self.draw.device.composited_here(node) {
            Some(true) => log::info!(
                target: "reims_vgpu::graphics",
                "the window is composited on DRM node {major}:{minor}, the runtime's GPU: the \
                 window presents from it"
            ),
            Some(false) => log::info!(
                target: "reims_vgpu::graphics",
                "the window is composited on DRM node {major}:{minor}, another GPU than the \
                 runtime's: the window presents from the runtime's GPU, each present crossing to \
                 the compositor's through the window system (the rt platform makes no window \
                 device there)"
            ),
            None => log::info!(
                target: "reims_vgpu::graphics",
                "the window is composited on DRM node {major}:{minor}, and the runtime's GPU names \
                 no DRM node: the window presents from the runtime's GPU"
            ),
        }
    }
}

impl FrameSink for RtScanoutSink {
    /// The window's surface on rt's instance and a swapchain over it on rt's device; none where
    /// either is refused, which is logged.
    fn presenter(&self, target: &dyn VulkanTarget) -> Option<Box<dyn Presenter>> {
        self.log_route(target);
        match swapchain_over(&self.draw, target) {
            Ok(swapchain) => Some(Box::new(RtPresenter {
                draw: Arc::clone(&self.draw),
                swapchain,
            })),
            Err(refused) => {
                log::warn!(
                    target: "reims_vgpu::graphics", kind:% = refused;
                    "no swapchain for the graphics display's window ({refused}): it stays empty"
                );
                None
            }
        }
    }
}

/// A swapchain over the window's surface `target` makes on the instance of `draw`'s device.
fn swapchain_over(
    draw: &ScanoutDraw<RtPlatform>,
    target: &dyn VulkanTarget,
) -> Result<RtSwapchain, Refused> {
    // SAFETY: the target makes the surface on the instance it is given, and hands it over.
    entry::run(|_| unsafe {
        RtSwapchain::over(&draw.device, |get_instance_proc_addr, instance| {
            target.create_surface(get_instance_proc_addr, instance)
        })
    })
}

/// A graphics window's presenter on the rt platform: the scanout's draw and the window's
/// swapchain, on its render thread.
struct RtPresenter {
    /// The scanout's buffer, the queue and the scanout.
    draw: Arc<ScanoutDraw<RtPlatform>>,
    /// The swapchain.
    swapchain: RtSwapchain,
}

impl Presenter for RtPresenter {
    /// The next image, the display drawn into it, with `cursor` drawn over it where the window
    /// draws the guest's cursor (`sink`'s module doc, "The guest's cursor"), and its present once
    /// rt has scheduled the draw; nothing, the image kept for the next draw, where the draw is
    /// refused (`sink`'s module doc, "A refused draw"), or rt refuses its present or its commit.
    fn draw(
        &mut self,
        drawable: (u64, u64),
        scale: f64,
        region: PixelRect,
        cursor: Option<&DrawnCursor>,
    ) -> Drawn {
        entry::run(|_| {
            self.draw.drawn(drawable, scale);
            let frame = match self.swapchain.acquire(drawable) {
                Acquired::Image(frame) => frame,
                Acquired::Retry => return Drawn::Retry,
                // rt parks no device: a take that ends answers Failed.
                Acquired::Failed => return Drawn::Ended,
            };
            let Some(texture) = self.swapchain.texture(frame) else {
                return Drawn::Nothing;
            };
            let swapchain = &self.swapchain;
            let committed = self.draw.scoped(|| {
                let (command_buffer, presents) = self.draw.encode(texture, region, cursor)?;
                // A draw dropped here answers its presents at the next refresh.
                if !swapchain.encode_ready(frame, &command_buffer) {
                    return None;
                }
                presents.after(&command_buffer);
                if window_times::on() {
                    // Its GPU time, for the host's summary (the module doc).
                    let ran = Box::new(|info: CompletedInfo| {
                        window_times::frame_ran(info.gpu_times_ns);
                    });
                    no_catch(command_buffer.add_completed_handler(OnFault::Die, ran));
                }
                no_catch(command_buffer.commit(OnFault::Die));
                Some(command_buffer)
            });
            let Some(command_buffer) = committed else {
                return Drawn::Nothing;
            };
            match self.swapchain.present(frame, &command_buffer) {
                true => Drawn::Presented,
                false => Drawn::Nothing,
            }
        })
    }
}
