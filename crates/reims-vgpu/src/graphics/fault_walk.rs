//! A decoder's tables, told to the rt platform's device as the decoder is made, which walks them
//! at its loss to place the driver's fault addresses among their live objects
//! (`reims_vgpu_vulkan::rt_platform::fault`'s docs). The walk keeps no record before the loss: it
//! reads the tables as they are then, waiting `fault::BOUND` in all at most for their locks, and
//! names the contexts whose tables it did not have in time. A device of another platform is told
//! nothing.

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
