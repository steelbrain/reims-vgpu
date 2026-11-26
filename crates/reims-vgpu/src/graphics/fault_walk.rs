//! A decoder's tables, told to the rt platform's device as the decoder is made, which walks them
//! at its loss to place the driver's fault addresses among their live objects
//! (`reims_vgpu_vulkan::rt_platform::fault`'s docs). The walk keeps no record before the loss: it
//! reads the tables as they are then, waiting `fault::BOUND` in all at most for their locks, and
//! names the contexts whose tables it did not have in time. Where the decoder goes during the
//! walk, so the walk holds its state last, the state is let go of on a thread of its own, not on
//! the walk's, which may be rt's completion thread (`decode::TableWalk::each`). A device of
//! another platform is told nothing.
//!
//! As a table stores a buffer, texture or heap, the object is told the guest's id for it
//! ([`name`]), which the device's history of released ranges and its lines of the command buffers
//! in flight name it by (`fault::Named`). As the host makes a buffer or texture of its own that a
//! display's draws use, or presents a guest's texture, the object is told what the host made it
//! for ([`name_role`]: the scanout's import, a presented surface's texture and the display's view
//! of it, the display colour's table and ramps, the guest cursor's glyph, the window overlay's
//! image, a host read's target), which those lines name it by where it has no guest id
//! (`fault::Role`).

use std::any::Any;
use std::sync::Arc;

use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::rt_platform::fault::{self, LiveObjects, Walked};
use reims_vgpu_vulkan::rt_platform::{RtDevice, RtObj, RtPlatform};

use super::decode::{TableObject, TableWalk};

/// The rt platform's decoder's tables, as its device walks them.
struct Tables(TableWalk<RtPlatform>);

impl LiveObjects for Tables {
    fn walk(&self) -> Walked {
        let mut walked = Walked::default();
        let (contexts, unwalked, busy) = self.0.each(
            fault::BOUND,
            &mut |context, id, gone, object| match object {
                TableObject::Buffer(buffer) => {
                    walked.add(fault::buffer_parts(context, id, gone, buffer), false);
                }
                TableObject::Texture(texture) => {
                    walked.add(fault::texture_parts(context, id, gone, texture), true);
                }
                TableObject::Heap(heap) => {
                    walked.add(fault::heap_parts(context, id, gone, heap), false);
                }
                TableObject::ManagedCopy(buffer) => {
                    walked.add(fault::managed_copy_parts(context, id, gone, buffer), false);
                }
            },
        );
        walked.contexts = contexts;
        walked.unwalked = unwalked;
        walked.contexts_busy = busy;
        walked
    }

    fn live(&self) -> bool {
        self.0.is_live()
    }
}

/// Tells `object`, which a decoder's table stores under `id`, that id (`fault::Named`), where the
/// platform is the rt platform: a downcast, which each platform's build settles, and an atomic
/// store, two for a `Managed` texture's CPU-copy buffer; nothing for any other platform.
pub(super) fn name<P: Platform>(id: u64, object: TableObject<'_, P>) {
    type Rt = RtPlatform;
    match object {
        TableObject::Buffer(buffer) => {
            if let Some(buffer) = (buffer as &dyn Any).downcast_ref::<<Rt as Platform>::Buffer>() {
                fault::name_buffer(buffer, id);
            }
        }
        TableObject::Texture(texture) => {
            let texture = (texture as &dyn Any).downcast_ref::<<Rt as Platform>::Texture>();
            if let Some(texture) = texture {
                fault::name_texture(texture, id);
            }
        }
        TableObject::Heap(heap) => {
            if let Some(heap) = (heap as &dyn Any).downcast_ref::<<Rt as Platform>::Heap>() {
                fault::name_heap(heap, id);
            }
        }
        TableObject::ManagedCopy(buffer) => {
            if let Some(buffer) = (buffer as &dyn Any).downcast_ref::<<Rt as Platform>::Buffer>() {
                fault::name_managed_copy(buffer, id);
            }
        }
    }
}

/// An object of the host's own, or a guest's it presents, to tell what the host made it for
/// ([`name_role`]).
pub(crate) enum HostObject<'a, P: Platform> {
    /// A buffer.
    Buffer(&'a P::Buffer),
    /// A texture.
    Texture(&'a P::Texture),
}

/// Tells `object` what the host made it for, or presents it as, `role` (`fault::Named`), where
/// the platform is the rt platform: a downcast, which each platform's build settles, and an atomic
/// store; nothing for any other platform.
pub(crate) fn name_role<P: Platform>(role: fault::Role, object: HostObject<'_, P>) {
    type Rt = RtPlatform;
    match object {
        HostObject::Buffer(buffer) => {
            if let Some(buffer) = (buffer as &dyn Any).downcast_ref::<<Rt as Platform>::Buffer>() {
                fault::name_role_buffer(buffer, role);
            }
        }
        HostObject::Texture(texture) => {
            let texture = (texture as &dyn Any).downcast_ref::<<Rt as Platform>::Texture>();
            if let Some(texture) = texture {
                fault::name_role_texture(texture, role);
            }
        }
    }
}

/// Tells `device` of the decoder's tables `walk`, where the platform is the rt platform; nothing
/// for any other.
pub(super) fn watch<P: Platform>(device: &P::Device, walk: TableWalk<P>) {
    let Some(device) = (device as &dyn Any).downcast_ref::<RtObj<RtDevice>>() else {
        return;
    };
    let walk: Box<dyn Any> = Box::new(walk);
    let Ok(walk) = walk.downcast::<TableWalk<RtPlatform>>() else {
        return;
    };
    device.watch_faults(Arc::new(Tables(*walk)));
}
