//! A graphics display's window sink on the Vulkan platform, on either OS: [`VulkanScanoutSink`]
//! makes a presenter over a swapchain of the window's surface, and each draw blits the scanout
//! into the swapchain's next image (its usage takes transfers) and presents it.
//!
//! **The compositor's GPU.** Where the host's compositor composites the window on another GPU
//! than the runtime's (the window system names it: [`VulkanTarget::compositor_device`]), every
//! image a swapchain of the runtime's device presents crosses that GPU's link to the
//! compositor's (docs/graphics-kext/design.md, Decided 14, "the window shown by the compositor's
//! GPU"). There the sink makes a device of the window's own on the compositor's GPU
//! (`VulkanPlatform::window_device`) and a draw on it ([`ScanoutDraw::on_compositor`]), whose
//! swapchain the presenter presents through: it draws the scanout from its guest pages and the
//! presented surfaces from the window's mirror in host memory (`mirror`), which the runtime's GPU
//! brings up to each presented frame sending only the tiles that changed, so the images are the
//! compositor's GPU's own and only the frames' changes cross the runtime GPU's link. The route
//! is asked once a sink, at its first presenter ([`route`]), and falls back to the runtime's
//! device, for good, at each refusal on the way, on every driver: no main device in the window
//! system's feedback, a main device that is the runtime's own GPU or no GPU the runtime makes
//! devices of, a device or a queue the runtime refuses there, a swapchain refused over the
//! window's surface, and a presented surface the compositor's GPU cannot draw (a format the
//! mirror does not carry, or the mirror, its sync, its import or its texture refused,
//! [`ScanoutDraw::foreign_refused`]), at which the presenter ends and the next is the runtime's
//! device's.

use std::sync::Arc;

use reims_vgpu_graphics::region::Region;
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::core::platform::command::CommandBuffer as _;
use reims_vgpu_vulkan::core::raise::OnFault;
use reims_vgpu_vulkan::core::sync::Mutex;
use reims_vgpu_vulkan::device_choice::{CompositorGpu, compositor_gpu};
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
    let draw = ScanoutDraw::new(Arc::clone(&platform), region, shown, presented)?;
    Some(Arc::new(VulkanScanoutSink {
        draw: Arc::new(draw),
        platform,
        region: Arc::clone(region),
        shown: Arc::clone(shown),
        presented: Arc::clone(presented),
        route: Arc::new(Mutex::new(Route::Unasked)),
    }))
}

/// Where the window presents from (the module doc, "The compositor's GPU").
enum Route<D> {
    /// Not asked yet: the sink's first presenter asks.
    Unasked,
    /// From the runtime's device.
    Own,
    /// From a device of the window's own on the compositor's GPU, through this draw.
    Compositor(Arc<D>),
}

impl<D> Clone for Route<D> {
    fn clone(&self) -> Self {
        match self {
            Route::Unasked => Route::Unasked,
            Route::Own => Route::Own,
            Route::Compositor(draw) => Route::Compositor(Arc::clone(draw)),
        }
    }
}

/// The route the window presents by, of the window system's main device `node` (`None`: it names
/// none), the GPU `gpu` answers for it against the runtime's own, a device `device` makes on
/// that GPU by its registry ID, and a draw `draw` makes on that device: the draw, or
/// [`Route::Own`] at the first of them that refuses (the module doc). Each step is logged once.
fn route<V, D>(
    node: Option<(u32, u32)>,
    gpu: impl FnOnce((u32, u32)) -> CompositorGpu,
    device: impl FnOnce(u64) -> Option<V>,
    draw: impl FnOnce(V) -> Option<D>,
) -> Route<D> {
    let Some(node) = node else {
        log::info!(
            target: "reims_vgpu::graphics",
            "the window system names no GPU the window is composited on: the window presents \
             from the runtime's GPU"
        );
        return Route::Own;
    };
    let id = match gpu(node) {
        CompositorGpu::Own => {
            log::info!(
                target: "reims_vgpu::graphics",
                "the window is composited on DRM node {}:{}, the runtime's GPU or one it makes \
                 no device of: the window presents from the runtime's GPU",
                node.0,
                node.1
            );
            return Route::Own;
        }
        CompositorGpu::Other(id) => id,
    };
    let Some(made) = device(id) else {
        return Route::Own;
    };
    match draw(made) {
        Some(draw) => {
            log::info!(
                target: "reims_vgpu::graphics",
                "the window is composited on DRM node {}:{}, another GPU than the runtime's \
                 (registry ID {id:#x}): the window presents from a device of its own there",
                node.0,
                node.1
            );
            Route::Compositor(Arc::new(draw))
        }
        None => {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the compositor's GPU (registry ID {id:#x}) gave the window's device no queue: \
                 the window presents from the runtime's GPU"
            );
            Route::Own
        }
    }
}

/// What presents a graphics display's scanout in its window on the Vulkan platform.
pub(super) struct VulkanScanoutSink {
    /// The scanout's buffer, the queue and the scanout on the runtime's device; shared with the
    /// presenters.
    draw: Arc<ScanoutDraw<VulkanPlatform>>,
    /// What a draw on the compositor's GPU is made of.
    platform: Arc<VulkanPlatform>,
    region: Arc<Region>,
    shown: Arc<Shown>,
    presented: Arc<Presented<VulkanPlatform>>,
    /// Where the window presents from, asked at the first presenter; shared with the presenters,
    /// which set it to [`Route::Own`] where the compositor's GPU cannot draw a surface.
    route: Arc<Mutex<Route<ScanoutDraw<VulkanPlatform>>>>,
}

impl VulkanScanoutSink {
    /// The route, asked of `target` where it is not yet (the module doc).
    fn route_for(&self, target: &dyn VulkanTarget) -> Route<ScanoutDraw<VulkanPlatform>> {
        let mut kept = self.route.lock();
        if let Route::Unasked = *kept {
            let made = route(
                target.compositor_device(),
                |node| compositor_gpu(node, VulkanPlatform::default_device_registry_id()),
                VulkanPlatform::window_device,
                |device| {
                    ScanoutDraw::on_compositor(
                        Arc::clone(&self.platform),
                        device,
                        &self.region,
                        &self.shown,
                        &self.presented,
                    )
                },
            );
            *kept = made;
        }
        kept.clone()
    }

    /// The window presents from the runtime's GPU from now, for good.
    fn fall_back(route: &Mutex<Route<ScanoutDraw<VulkanPlatform>>>) {
        *route.lock() = Route::Own;
    }
}

impl FrameSink for VulkanScanoutSink {
    /// The window's surface on the instance of the device the window presents from (the module
    /// doc) and a swapchain over it, on the draw's queue's device; none where either is refused,
    /// which is logged. A swapchain refused on the compositor's GPU falls back to the runtime's
    /// device, for good.
    fn presenter(&self, target: &dyn VulkanTarget) -> Option<Box<dyn Presenter>> {
        let draw = match entry::run(|_| self.route_for(target)) {
            Route::Compositor(draw) => {
                match swapchain_over(&draw, target) {
                    Ok(swapchain) => {
                        return Some(Box::new(ScanoutPresenter {
                            draw,
                            swapchain,
                            route: Some(Arc::clone(&self.route)),
                        }));
                    }
                    Err(refused) => {
                        log::warn!(
                            target: "reims_vgpu::graphics", kind:% = refused;
                            "no swapchain for the graphics display's window on the compositor's \
                             GPU ({refused}): the window presents from the runtime's GPU"
                        );
                        Self::fall_back(&self.route);
                    }
                }
                Arc::clone(&self.draw)
            }
            Route::Own | Route::Unasked => Arc::clone(&self.draw),
        };
        match swapchain_over(&draw, target) {
            Ok(swapchain) => Some(Box::new(ScanoutPresenter {
                draw,
                swapchain,
                route: None,
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

/// A swapchain over the window's surface `target` makes on the instance of `draw`'s queue's
/// device.
fn swapchain_over(
    draw: &ScanoutDraw<VulkanPlatform>,
    target: &dyn VulkanTarget,
) -> Result<Swapchain, reims_vgpu_vulkan::swapchain::Refused> {
    // SAFETY: the target makes the surface on the instance it is given, and hands it over.
    entry::run(|_| unsafe {
        Swapchain::over(&draw.queue, |get_instance_proc_addr, instance| {
            target.create_surface(get_instance_proc_addr, instance)
        })
    })
}

/// A graphics window's presenter: the scanout's draw and the window's swapchain, on its render
/// thread.
struct ScanoutPresenter {
    /// The scanout's buffer, the queue and the scanout.
    draw: Arc<ScanoutDraw<VulkanPlatform>>,
    /// The swapchain.
    swapchain: Swapchain,
    /// The sink's route, where the draw is on the compositor's GPU: set to the runtime's device
    /// where the draw meets a surface it cannot draw there.
    route: Option<Arc<Mutex<Route<ScanoutDraw<VulkanPlatform>>>>>,
}

impl Presenter for ScanoutPresenter {
    /// The next image, the scanout blitted into it, with `cursor` drawn over it where the window
    /// draws the guest's cursor (`sink`'s module doc, "The guest's cursor"), and its present once
    /// the runtime has scheduled the draw; nothing, the image kept for the next draw, where the
    /// draw is refused (`sink`'s module doc, "A refused draw").
    /// On the compositor's GPU, a draw that meets a surface it cannot draw there ends the
    /// presenter, and the next is the runtime's device's (the module doc).
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
                Acquired::Failed => return Drawn::Ended,
                Acquired::Parked => return Drawn::Parked,
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
                no_catch(command_buffer.commit(OnFault::Die));
                Some(command_buffer)
            });
            let Some(command_buffer) = committed else {
                if let Some(route) = &self.route
                    && self.draw.foreign_refused()
                {
                    VulkanScanoutSink::fall_back(route);
                    return Drawn::Ended;
                }
                return Drawn::Nothing;
            };
            self.swapchain.present(frame, command_buffer);
            Drawn::Presented
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The route of a node, a GPU's answer for it, and whether the device and the draw are made:
    /// the draw's value where it is made.
    fn routed(
        node: Option<(u32, u32)>,
        gpu: CompositorGpu,
        device: bool,
        draw: bool,
    ) -> Option<u64> {
        let route = route(
            node,
            |_| gpu,
            |id| device.then_some(id),
            |id| draw.then_some(id),
        );
        match route {
            Route::Compositor(made) => Some(*made),
            Route::Own | Route::Unasked => None,
        }
    }

    /// The window presents from a device of its own only where the window system names a main
    /// device, it is another GPU than the runtime's, and the runtime makes a device and a queue
    /// there: each refusal on the way falls back to the runtime's GPU, whatever the driver
    #[test]
    fn the_window_presents_from_the_compositors_gpu_only_where_nothing_refuses() {
        let node = Some((226, 129));
        let other = CompositorGpu::Other(7);
        assert_eq!(routed(node, other, true, true), Some(7));
        // No dma-buf feedback (X11, an older compositor).
        assert_eq!(routed(None, other, true, true), None);
        // The compositor composites on the runtime's GPU, or on a GPU the runtime cannot use.
        assert_eq!(routed(node, CompositorGpu::Own, true, true), None);
        // The runtime refuses a device on the compositor's GPU.
        assert_eq!(routed(node, other, false, true), None);
        // The device gives the draw no queue.
        assert_eq!(routed(node, other, true, false), None);
    }
}
