//! A graphics display's window sink on the product's platform: [`MetalScanoutSink`] blits the
//! scanout straight into each drawable the window's render thread takes of its layer, and
//! presents it in the same command buffer.

use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer as _, MTLDevice, MTLDrawable};
use objc2_quartz_core::CAMetalDrawable;
use reims_graphics::region::Region;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::command::CommandBuffer as _;
use reims_vgpu_core::platform::resource::Texture as _;
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_metal::{MetalPlatform, Obj, msg};
use reims_vgpu_window::{DrawnCursor, FrameSink, PixelRect};

use super::sink::ScanoutDraw;
use super::{Presented, Shown};
use crate::entry;
use crate::entry::no_catch;

/// The sink of a graphics display on Metal, over `region`; `None` where Metal refuses a queue.
pub(super) fn sink(
    platform: Arc<MetalPlatform>,
    region: &Arc<Region>,
    shown: &Arc<Shown>,
    presented: &Arc<Presented<MetalPlatform>>,
) -> Option<Arc<dyn FrameSink>> {
    let draw = ScanoutDraw::new(platform, region, shown, presented)?;
    Some(Arc::new(MetalScanoutSink { draw }))
}

/// What blits a graphics display's scanout into its window's drawables on Metal.
struct MetalScanoutSink {
    /// The scanout's buffer, the queue and the scanout.
    draw: ScanoutDraw<MetalPlatform>,
}

impl FrameSink for MetalScanoutSink {
    /// The default Metal device, which the queue and the buffer are on.
    fn device(&self) -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
        Some(self.draw.device.as_retained().clone())
    }

    /// The scanout blitted into `drawable`'s texture, presented in the same command buffer;
    /// false, the drawable dropped, where the guest has no scanout, or where the draw is refused
    /// (`sink`'s module doc, "A refused draw"). The window's cursor is the host's, so `cursor` is
    /// never given (`graphics`'s module doc).
    fn draw(
        &self,
        drawable: &ProtocolObject<dyn CAMetalDrawable>,
        region: PixelRect,
        _cursor: Option<&DrawnCursor>,
    ) -> bool {
        entry::run(|_| {
            let texture: <MetalPlatform as Platform>::Texture =
                Obj::from_retained(msg::get("texture", || drawable.texture()));
            self.draw.drawn((texture.width(), texture.height()));
            self.draw
                .scoped(|| {
                    let (command_buffer, presents) = self.draw.encode(&texture, region)?;
                    let drawable: &ProtocolObject<dyn MTLDrawable> =
                        ProtocolObject::from_ref(drawable);
                    msg::get("presentDrawable:", || {
                        command_buffer.get().presentDrawable(drawable);
                    });
                    presents.after(&command_buffer);
                    no_catch(command_buffer.commit(OnFault::Die));
                    Some(())
                })
                .is_some()
        })
    }

    /// False: the scanout is blitted into the drawables, which must take copies.
    fn render_target_only(&self) -> bool {
        false
    }
}
